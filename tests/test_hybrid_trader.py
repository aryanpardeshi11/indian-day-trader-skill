"""Comprehensive unit & integration tests for nse-bse-hybrid-trader."""

from __future__ import annotations

import datetime
import gzip
import http.client
import io
import json
import sqlite3
import urllib.error
import urllib.request
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import indian_risk_gate
import nse_data_loader
import nse_timemachine
import smartapi_bridge
from broker_bridge import (
    close_paper_trade as mb_close_paper_trade,
)
from broker_bridge import (
    list_supported_brokers,
)
from broker_bridge import (
    lookup_scrip as mb_lookup_scrip,
)
from broker_bridge import (
    read_ledger as mb_read_ledger,
)
from broker_bridge import (
    stage_or_execute_order as mb_stage_or_execute_order,
)
from broker_bridge import (
    sync_scrip_master as mb_sync_scrip_master,
)
from indian_risk_gate import (
    calculate_indian_charges,
    compute_verdict_fingerprint,
    evaluate_risk_gate,
)
from nse_common import (
    EPOCH,
    IST_OFFSET,
    SUPPORTED_BROKERS,
    NSETraderError,
    normalize_broker,
    normalize_symbol,
    nse_session_info,
    parse_now_ist,
    round_tick,
)
from nse_data_loader import analyze_symbol, format_human_report
from nse_timemachine import run_timemachine
from smartapi_bridge import (
    _angel_rest_post,
    close_paper_trade,
    generate_totp,
    lookup_scrip,
    read_ledger,
    stage_or_execute_order,
    sync_scrip_master,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ist_to_unix(dt_ist: datetime.datetime) -> int:
    utc = (dt_ist - IST_OFFSET).replace(tzinfo=datetime.timezone.utc)
    return int((utc - EPOCH).total_seconds())


def _make_offline_nse_5m(path: Path, num_days: int = 30, base_price: float = 1000.0, drift: float = 0.0004) -> Path:
    """Create a realistic synthetic 75-bar/day NSE Yahoo v8 chart JSON."""
    stamps, opens, highs, lows, closes, vols = [], [], [], [], [], []
    start_day = datetime.date(2026, 9, 1)
    p = base_price
    days_made = 0
    cur_day = start_day
    while days_made < num_days:
        if cur_day.weekday() < 5:
            for slot in range(75):
                bar_dt = datetime.datetime.combine(cur_day, datetime.time(9, 15)) + datetime.timedelta(minutes=5 * slot)
                stamps.append(_ist_to_unix(bar_dt))
                o = p
                # Mild upward drift + realistic oscillation
                step = drift + (0.0012 if slot % 3 != 0 else -0.0006)
                c = round(o * (1.0 + step), 2)
                hi = round(max(o, c) * 1.0018, 2)
                lo = round(min(o, c) * 0.9988, 2)
                opens.append(o)
                highs.append(hi)
                lows.append(lo)
                closes.append(c)
                vols.append(50000 + slot * 500)
                p = c
            days_made += 1
        cur_day += datetime.timedelta(days=1)

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "RELIANCE.NS", "exchangeName": "NSI", "currency": "INR"},
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": vols,
                            }
                        ]
                    },
                }
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_offline_daily(path: Path, num_days: int = 220, base_price: float = 1100.0) -> Path:
    stamps, opens, highs, lows, closes, vols = [], [], [], [], [], []
    cur = datetime.date(2026, 1, 1)
    p = base_price
    made = 0
    while made < num_days:
        if cur.weekday() < 5:
            bar_dt = datetime.datetime.combine(cur, datetime.time(9, 15))
            stamps.append(_ist_to_unix(bar_dt))
            o = p
            delta = 0.004 if (made % 5 != 0) else -0.006
            c = round(o * (1.0 + delta), 2)
            hi = round(max(o, c) * 1.008, 2)
            lo = round(min(o, c) * 0.992, 2)
            opens.append(o)
            highs.append(hi)
            lows.append(lo)
            closes.append(c)
            vols.append(2_000_000 + (made % 10) * 100_000)
            p = c
            made += 1
        cur += datetime.timedelta(days=1)

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "RELIANCE.NS", "exchangeName": "NSI", "currency": "INR"},
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": vols,
                            }
                        ]
                    },
                }
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_symbol_normalization_and_tick_rounding() -> None:
    assert normalize_symbol("RELIANCE")["yahoo"] == "RELIANCE.NS"
    assert normalize_symbol("RELIANCE.BO")["exchange"] == "BSE"
    assert normalize_symbol("NIFTY")["yahoo"] == "^NSEI"
    assert normalize_symbol("BANKNIFTY")["yahoo"] == "^NSEBANK"
    assert normalize_symbol("SENSEX")["yahoo"] == "^BSESN"

    assert round_tick(785.12, mode="nearest") == 785.10
    assert round_tick(785.13, mode="nearest") == 785.15
    assert round_tick(778.04, mode="down") == 778.00
    assert round_tick(778.01, mode="up") == 778.05


def test_ist_session_windows_and_holidays() -> None:
    assert nse_session_info(parse_now_ist("2026-11-09 09:05"))["phase"] == "PRE_OPEN"
    assert nse_session_info(parse_now_ist("2026-11-09 09:20"))["phase"] == "OPENING_RANGE"
    assert nse_session_info(parse_now_ist("2026-11-09 10:15"))["intraday_entry_allowed"] is True
    assert nse_session_info(parse_now_ist("2026-11-09 14:35"))["phase"] == "HAZARD_WINDOW_NO_NEW_ENTRY"
    assert nse_session_info(parse_now_ist("2026-11-09 14:35"))["intraday_entry_allowed"] is False
    # Nov 10, 2026 is Diwali-Balipratipada NSE holiday
    hol = nse_session_info(parse_now_ist("2026-11-10 10:15"))
    assert hol["is_holiday"] is True
    assert hol["phase"] == "CLOSED_HOLIDAY"


def test_indian_charges_2026_segments() -> None:
    # EQ_INTRADAY: STT 0.025% sell, Stamp 0.003% buy
    intra = calculate_indian_charges(qty=50, entry_price=1000.0, exit_price=1020.0, side="buy", segment="EQ_INTRADAY")
    assert intra["buy_turnover"] == 50000.0
    assert intra["sell_turnover"] == 51000.0
    assert intra["brokerage"] == 30.30  # min(20, 15.0) + min(20, 15.3)
    assert intra["stt"] == 12.75       # 51000 * 0.00025
    assert intra["stamp_duty"] == 1.50 # 50000 * 0.00003
    assert intra["total_charges"] > 50.0

    # FNO_FUT April 2026 STT 0.05% on sell
    fut = calculate_indian_charges(qty=50, entry_price=24000.0, exit_price=24100.0, side="buy", segment="FNO_FUT")
    assert fut["stt"] == round(50 * 24100.0 * 0.0005, 2)
    assert fut["brokerage"] == 40.0

    # FNO_OPT April 2026 STT 0.15% on sell premium
    opt = calculate_indian_charges(qty=75, entry_price=200.0, exit_price=240.0, side="buy", segment="FNO_OPT")
    assert opt["stt"] == round(75 * 240.0 * 0.0015, 2)


def test_risk_gate_sizing_and_circuit_breakers(tmp_path: Path) -> None:
    # Valid intraday trade during primary window
    res = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=785.0,
        stop=778.0,
        target=800.0,
        segment="EQ_INTRADAY",
        symbol="TATAMOTORS",
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert res["verdict"] == "PASS"
    assert res["qty"] > 0
    assert res["net_loss_at_stop_inr"] <= 500.0
    assert res["net_rr"] >= 1.5
    assert res["fingerprint"] == compute_verdict_fingerprint(res)

    # Hard block when > 2.0% risk ceiling
    over_risk = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=2.5,
        entry=785.0,
        stop=778.0,
        target=800.0,
        segment="EQ_INTRADAY",
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert over_risk["verdict"] == "BLOCK"
    assert any("RISK_CAP_EXCEEDED" in b for b in over_risk["blocks"])

    # Hard block after 14:30 IST hazard window
    hazard = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=785.0,
        stop=778.0,
        target=800.0,
        segment="EQ_INTRADAY",
        now_str="2026-11-09 14:40",
        state_dir=tmp_path,
    )
    assert hazard["verdict"] == "BLOCK"
    assert any("INTRA_HAZARD_WINDOW" in b for b in hazard["blocks"])

    # Hard block near upper circuit (within 1%)
    uc = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=995.0,
        stop=985.0,
        target=1015.0,
        segment="EQ_INTRADAY",
        upper_circuit=1000.0,
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert uc["verdict"] == "BLOCK"
    assert any("UPPER_CIRCUIT_LOCK_RISK" in b for b in uc["blocks"])

    # Hard block on F&O Ban
    ban = evaluate_risk_gate(
        capital_inr=500000.0,
        risk_pct=1.0,
        entry=200.0,
        stop=180.0,
        target=245.0,
        segment="FNO_OPT",
        lot_size=75,
        fno_ban=True,
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert ban["verdict"] == "BLOCK"
    assert any("FNO_BAN" in b for b in ban["blocks"])


def test_data_loader_and_timemachine_offline(tmp_path: Path) -> None:
    f_5m = _make_offline_nse_5m(tmp_path / "RELIANCE.NS-5m.json", num_days=30, drift=0.0005)
    f_1d = _make_offline_daily(tmp_path / "RELIANCE.NS-1d.json", num_days=220)

    analysis = analyze_symbol(
        symbol="RELIANCE.NS",
        now_str="2026-11-09 11:00",
        offline_5m=str(f_5m),
        offline_1d=str(f_1d),
        offline_1wk=str(f_1d),
        delivery_pct=58.5,
    )
    assert analysis["symbol"]["yahoo"] == "RELIANCE.NS"
    assert analysis["intraday"]["bars_count_5m"] == 75
    assert analysis["intraday"]["anchored_vwap"] is not None
    assert analysis["daily"]["ema200"] is not None
    assert analysis["confluence"]["score_out_of"] == 6

    tm = run_timemachine(
        symbol="RELIANCE.NS",
        entry=1000.0,
        stop=990.0,
        target=1012.0,
        mode="INTRADAY",
        entry_time="09:30",
        until_ist="15:15",
        replays=2000,
        seed=19,
        offline_path=str(f_5m),
    )
    assert tm["replays"] == 2000
    assert tm["context"]["full_sessions_in_pool"] == 30
    assert tm["context"]["cutoff_ist"] == "15:15"
    assert 0.0 <= tm["target_first_resolved_pct"] <= 100.0

    # Swing mode using offline 1d
    tm_sw = run_timemachine(
        symbol="RELIANCE.NS",
        entry=1100.0,
        stop=1050.0,
        target=1180.0,
        mode="SWING",
        replays=1000,
        offline_path=str(f_1d),
    )
    assert tm_sw["mode"] == "SWING"
    assert tm_sw["context"]["swing_horizon_days"] == 10


def test_smartapi_scrip_cache_totp_and_paper_ledger(tmp_path: Path) -> None:
    # RFC-6238 official test vector check ("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ" @ t=59 -> 287082)
    assert generate_totp("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", at_time=59) == "94287082"[-6:]

    scrip_file = tmp_path / "OpenAPIScripMaster.json"
    scrip_file.write_text(
        json.dumps(
            [
                {
                    "token": "2885",
                    "symbol": "RELIANCE-EQ",
                    "name": "RELIANCE",
                    "expiry": "",
                    "strike": "-1",
                    "lotsize": "1",
                    "instrumenttype": "",
                    "exch_seg": "NSE",
                    "tick_size": "5.000000",
                },
                {
                    "token": "35000",
                    "symbol": "NIFTY26NOV24500CE",
                    "name": "NIFTY",
                    "expiry": "26NOV2026",
                    "strike": "2450000",
                    "lotsize": "75",
                    "instrumenttype": "OPTIDX",
                    "exch_seg": "NFO",
                    "tick_size": "5.000000",
                },
            ]
        ),
        encoding="utf-8",
    )
    sync_res = sync_scrip_master(offline_json=scrip_file, state_dir=tmp_path)
    assert sync_res["instruments"] == 2

    lk = lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path)
    assert lk is not None
    assert lk["token"] == "2885"
    assert lk["lookup_ms"] < 50.0

    # Stage paper trade & prevent live execution without CONFIRM LIVE
    staged = stage_or_execute_order(
        symbol="RELIANCE",
        entry=1324.0,
        stop=1310.0,
        target=1352.0,
        segment="EQ_INTRADAY",
        capital_inr=50000.0,
        risk_pct=1.0,
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert staged["mode"] == "PAPER"
    assert staged["status"] == "OPEN"
    assert staged["smartapi_payload"]["variety"] == "ROBO"
    assert staged["smartapi_payload"]["symboltoken"] == "2885"

    with pytest.raises(Exception, match="CONFIRM LIVE"):
        stage_or_execute_order(
            symbol="RELIANCE",
            entry=1324.0,
            stop=1310.0,
            target=1352.0,
            segment="EQ_INTRADAY",
            live=True,
            confirm_phrase="yes",
            now_str="2026-11-09 10:15",
            state_dir=tmp_path,
        )

    # Close trade with loss breaching 3% daily drawdown (-1500 INR), then confirm risk gate locks out next trade!
    closed = close_paper_trade(
        trade_id=staged["trade_id"],
        exit_price=1270.0,
        reason="STOP",
        now_str="2026-11-09 11:30",
        state_dir=tmp_path,
    )
    assert closed["realized_net_pnl_inr"] < -1500.0

    ledger = read_ledger(date_str="2026-11-09", state_dir=tmp_path)
    assert ledger["count"] == 1
    assert ledger["realized_net_pnl_inr"] < -1500.0

    blocked_next = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=1324.0,
        stop=1310.0,
        target=1352.0,
        segment="EQ_INTRADAY",
        symbol="RELIANCE",
        now_str="2026-11-09 11:45",
        state_dir=tmp_path,
    )
    assert blocked_next["verdict"] == "BLOCK"
    assert any("DAILY_DRAWDOWN_LOCK" in b for b in blocked_next["blocks"])


def test_cli_end_to_end_pipeline(tmp_path: Path) -> None:
    # Strong bullish upward drift so Time Machine passes >55% Target-First and positive EV
    f_5m = _make_offline_nse_5m(tmp_path / "SBIN.NS-5m.json", num_days=25, base_price=800.0, drift=0.0012)
    f_1d = _make_offline_daily(tmp_path / "SBIN.NS-1d.json", num_days=220, base_price=700.0)

    rc_dl = nse_data_loader.main(
        [
            "--symbol", "SBIN",
            "--now", "2026-11-09 10:15",
            "--offline-5m", str(f_5m),
            "--offline-1d", str(f_1d),
            "--offline-1wk", str(f_1d),
            "--json",
        ]
    )
    assert rc_dl == 0

    rc_tm = nse_timemachine.main(
        [
            "--symbol", "SBIN.NS",
            "--entry", "800",
            "--stop", "794",
            "--target", "812",
            "--mode", "INTRADAY",
            "--entry-time", "09:30",
            "--replays", "1000",
            "--offline", str(f_5m),
            "--state-dir", str(tmp_path),
            "--write",
            "--json",
        ]
    )
    assert rc_tm == 0
    tm_json = tmp_path / "timemachine_result.json"
    assert tm_json.exists()

    rc_rg = indian_risk_gate.main(
        [
            "--symbol", "SBIN",
            "--capital", "50000",
            "--risk-pct", "1.0",
            "--entry", "800",
            "--stop", "794",
            "--target", "812",
            "--segment", "EQ_INTRADAY",
            "--timemachine-json", str(tm_json),
            "--now", "2026-11-09 10:15",
            "--state-dir", str(tmp_path),
            "--write",
            "--json",
        ]
    )
    assert rc_rg == 0
    risk_json = tmp_path / "risk_verdict.json"
    assert risk_json.exists()

    rc_st = smartapi_bridge.main(
        [
            "stage",
            "--symbol", "SBIN",
            "--segment", "EQ_INTRADAY",
            "--entry", "800",
            "--stop", "794",
            "--target", "812",
            "--risk-json", str(risk_json),
            "--now", "2026-11-09 10:15",
            "--state-dir", str(tmp_path),
            "--paper",
            "--json",
        ]
    )
    assert rc_st == 0

    # Verify fingerprint tamper protection rejects modified risk_verdict.json
    tampered = json.loads(risk_json.read_text(encoding="utf-8"))
    tampered["qty"] = 9999
    risk_json.write_text(json.dumps(tampered), encoding="utf-8")
    rc_tampered = smartapi_bridge.main(
        [
            "stage",
            "--symbol", "SBIN",
            "--segment", "EQ_INTRADAY",
            "--entry", "800",
            "--stop", "794",
            "--target", "812",
            "--risk-json", str(risk_json),
            "--now", "2026-11-09 10:15",
            "--state-dir", str(tmp_path),
            "--paper",
        ]
    )
    assert rc_tampered == 1


def test_timemachine_dual_gate_modes(tmp_path: Path) -> None:
    # Moderate bullish drift (~0.00025/bar) so 1.8R passes r_breakeven (~38-42%) but fails strict_55 (55%)
    f_5m = _make_offline_nse_5m(tmp_path / "INFY.NS-5m.json", num_days=25, base_price=1500.0, drift=0.0001)
    tm_be = run_timemachine(
        symbol="INFY.NS",
        entry=1500.0,
        stop=1490.0,
        target=1518.0,
        mode="INTRADAY",
        entry_time="09:30",
        replays=2000,
        seed=19,
        gate_mode="r_breakeven",
        offline_path=str(f_5m),
    )
    assert tm_be["gate_mode"] == "r_breakeven"
    assert tm_be["breakeven_prob_net_pct"] < tm_be["strict_prob_floor_pct"]

    # Force a synthetic borderline Time Machine result (48% hit rate @ 1.8R, +0.25R EV) to verify gate split
    borderline_tm = {
        "target_first_resolved_pct": 48.0,
        "r_at_target": 1.8,
        "estimated_tax_drag_r": 0.10,
        "breakeven_prob_net_pct": 39.29,
        "avg_net_r": 0.25,
    }
    pass_be = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=1500.0,
        stop=1490.0,
        target=1520.0,
        segment="EQ_INTRADAY",
        symbol="INFY",
        timemachine_result=borderline_tm,
        tm_gate_mode="r_breakeven",
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert pass_be["verdict"] == "PASS"

    block_55 = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=1500.0,
        stop=1490.0,
        target=1520.0,
        segment="EQ_INTRADAY",
        symbol="INFY",
        timemachine_result=borderline_tm,
        tm_gate_mode="strict_55",
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert block_55["verdict"] == "BLOCK"
    assert any("TIMEMACHINE_PROB_LOW" in b for b in block_55["blocks"])

    # Verify fallback safety when avg_net_r is explicitly None
    none_ev_tm = {
        "target_first_resolved_pct": 52.0,
        "r_at_target": 1.8,
        "estimated_tax_drag_r": 0.10,
        "breakeven_prob_net_pct": 39.29,
        "avg_net_r": None,
        "avg_gross_r": 0.18,
    }
    fallback_ev = evaluate_risk_gate(
        capital_inr=50000.0,
        risk_pct=1.0,
        entry=1500.0,
        stop=1490.0,
        target=1520.0,
        segment="EQ_INTRADAY",
        symbol="INFY",
        timemachine_result=none_ev_tm,
        tm_gate_mode="r_breakeven",
        now_str="2026-11-09 10:15",
        state_dir=tmp_path,
    )
    assert fallback_ev["verdict"] == "PASS"


class _DummyHTTPResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _DummyHTTPResp:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def test_angel_rest_post_error_handling_and_validation() -> None:
    url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword"

    # 1. Valid JSON object response
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResp(b'{"status": true, "data": {"jwtToken": "tok"}}'),
    ):
        res = _angel_rest_post(url, {"clientcode": "X"}, {"X-PrivateKey": "k"})
        assert res["status"] is True
        assert res["data"]["jwtToken"] == "tok"

    # 2. HTTPError preserves SmartAPI JSON message/errorcode and closes fp
    err_fp = io.BytesIO(b'{"status": false, "message": "Invalid totp", "errorcode": "AB1050"}')
    http_err = urllib.error.HTTPError(
        url,
        400,
        "Bad Request",
        http.client.HTTPMessage(),
        err_fp,
    )
    with (
        patch("urllib.request.urlopen", side_effect=http_err),
        pytest.raises(NSETraderError, match=r"HTTP 400 \[AB1050\]: Invalid totp"),
    ):
        _angel_rest_post(url, {"clientcode": "X"}, {})
    assert err_fp.closed

    # 3. URLError / OSError network failure raises NSETraderError
    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")),
        pytest.raises(NSETraderError, match="timed out"),
    ):
        _angel_rest_post(url, {"clientcode": "X"}, {})

    # 4. http.client.HTTPException (e.g. BadStatusLine) raises NSETraderError
    with (
        patch("urllib.request.urlopen", side_effect=http.client.BadStatusLine("HTTP/1.1 ?")),
        pytest.raises(NSETraderError, match="Angel One SmartAPI HTTP failure"),
    ):
        _angel_rest_post(url, {"clientcode": "X"}, {})

    # 5. Non-dict JSON body rejected safely
    with (
        patch("urllib.request.urlopen", return_value=_DummyHTTPResp(b'["unexpected"]')),
        pytest.raises(NSETraderError, match="expected an object"),
    ):
        _angel_rest_post(url, {"clientcode": "X"}, {})

    # 6. Non-HTTP/HTTPS scheme blocked before Request/urlopen
    with (
        patch("urllib.request.urlopen") as mock_open,
        pytest.raises(NSETraderError, match="Unsupported URL scheme"),
    ):
        _angel_rest_post("file:///etc/passwd", {"clientcode": "X"}, {})
    mock_open.assert_not_called()


def test_multi_broker_charge_profiles_and_normalization() -> None:
    assert normalize_broker("kite") == "zerodha"
    assert normalize_broker("smartapi") == "angel_one"
    assert normalize_broker("breeze") == "icici_breeze"
    assert normalize_broker("finvasia") == "shoonya"
    assert normalize_broker("kotak") == "kotak_neo"
    with pytest.raises(NSETraderError, match="Unsupported broker"):
        normalize_broker("unknown_broker")

    brokers_info = list_supported_brokers()
    assert brokers_info["count"] == len(SUPPORTED_BROKERS) == 9

    # Zero brokerage on Shoonya vs Rs. 10 cap/leg on Kotak Neo vs Rs. 20 cap/leg on Zerodha
    shoonya_intra = calculate_indian_charges(50, 1000.0, 1020.0, "buy", "EQ_INTRADAY", broker="shoonya")
    kotak_intra = calculate_indian_charges(50, 1000.0, 1020.0, "buy", "EQ_INTRADAY", broker="kotak_neo")
    zerodha_del = calculate_indian_charges(50, 1000.0, 1020.0, "buy", "EQ_DELIVERY", broker="zerodha")
    upstox_del = calculate_indian_charges(50, 1000.0, 1020.0, "buy", "EQ_DELIVERY", broker="upstox")
    angel_small = calculate_indian_charges(1, 500.0, 510.0, "buy", "EQ_INTRADAY", broker="angel_one")

    assert shoonya_intra["brokerage"] == 0.0
    assert kotak_intra["brokerage"] == 20.0  # Rs. 10 per leg
    assert zerodha_del["brokerage"] == 0.0
    assert upstox_del["brokerage"] == 40.0  # Rs. 20 per leg
    assert angel_small["brokerage"] == 10.0  # min Rs. 5 per leg floor


def test_multi_broker_scrip_sync_payloads_and_live_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Zerodha CSV instrument sync & lookup
    kite_csv = tmp_path / "zerodha.csv"
    kite_csv.write_text(
        "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange\n"
        "738561,2885,RELIANCE,RELIANCE,0,,0,0.05,1,EQ,NSE,NSE\n",
        encoding="utf-8",
    )
    assert mb_sync_scrip_master(offline_json=kite_csv, state_dir=tmp_path, broker="zerodha")["instruments"] == 1
    z_lk = mb_lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path, broker="zerodha")
    assert z_lk is not None
    assert z_lk["token"] == "738561"

    # 2. Upstox .json.gz instrument sync & lookup
    upstox_gz = tmp_path / "upstox.json.gz"
    upstox_gz.write_bytes(
        gzip.compress(
            json.dumps(
                [
                    {
                        "segment": "NSE_EQ",
                        "exchange": "NSE",
                        "instrument_key": "NSE_EQ|INE002A01018",
                        "trading_symbol": "RELIANCE",
                        "name": "RELIANCE INDUSTRIES LTD",
                        "lot_size": 1,
                        "tick_size": 5.0,
                        "instrument_type": "EQ",
                    }
                ]
            ).encode("utf-8")
        )
    )
    assert mb_sync_scrip_master(offline_json=upstox_gz, state_dir=tmp_path, broker="upstox")["instruments"] == 1
    up_lk = mb_lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path, broker="upstox")
    assert up_lk is not None
    assert up_lk["token"] == "NSE_EQ|INE002A01018"

    # 3. Dhan CSV & Fyers 21-column headerless CSV & Groww CSV
    dhan_csv = tmp_path / "dhan.csv"
    dhan_csv.write_text(
        "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_TRADING_SYMBOL,SEM_CUSTOM_SYMBOL,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,SEM_LOT_UNITS,SEM_TICK_SIZE,SEM_INSTRUMENT_NAME\n"
        "NSE,E,2885,RELIANCE,RELIANCE,,0,1,0.05,EQUITY\n",
        encoding="utf-8",
    )
    assert mb_sync_scrip_master(offline_json=dhan_csv, state_dir=tmp_path, broker="dhan")["instruments"] == 1
    dh_lk = mb_lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path, broker="dhan")
    assert dh_lk is not None
    assert dh_lk["token"] == "2885"

    fyers_csv = tmp_path / "NSE_CM.csv"
    fyers_csv.write_text(
        "10100000002885,RELIANCE INDUSTRIES LTD,0,1,0.05,INE002A01018,0915-1530|1815-1915:,2026-11-09,,NSE:RELIANCE-EQ,10,10,2885,RELIANCE,2885,-1.0,XX,10100000002885,None,0,0.0\n",
        encoding="utf-8",
    )
    assert mb_sync_scrip_master(offline_json=fyers_csv, state_dir=tmp_path, broker="fyers")["instruments"] == 1
    fy_lk = mb_lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path, broker="fyers")
    assert fy_lk is not None
    assert fy_lk["symbol"] == "NSE:RELIANCE-EQ"

    fyers_fo_csv = tmp_path / "NSE_FO.csv"
    fyers_fo_csv.write_text(
        "1011261124500CE,NIFTY 26 NOV 24500 CE,0,75,0.05,,0915-1530:,2026-11-26,1795600000,NSE:NIFTY26NOV24500CE,10,10,35000,NIFTY,35000,24500,CE,10100000002600,None,0,0.0\n",
        encoding="utf-8",
    )
    assert mb_sync_scrip_master(offline_json=fyers_fo_csv, state_dir=tmp_path, broker="fyers")["instruments"] == 1
    fy_fo_lk = mb_lookup_scrip("NIFTY26NOV24500CE", exchange="NFO", state_dir=tmp_path, broker="fyers")
    assert fy_fo_lk is not None
    assert fy_fo_lk["lotsize"] == 75

    groww_csv = tmp_path / "groww.csv"
    groww_csv.write_text(
        "exchange,exchange_token,trading_symbol,groww_symbol,name,instrument_type,segment,series,isin,underlying_symbol,underlying_exchange_token,expiry_date,strike_price,lot_size,tick_size,freeze_quantity,is_reserved,buy_allowed,sell_allowed\n"
        "NSE,2885,RELIANCE,NSE-RELIANCE,RELIANCE INDUSTRIES LTD,EQ,CASH,EQ,INE002A01018,RELIANCE,,,,1,0.05,100000,False,True,True\n",
        encoding="utf-8",
    )
    assert mb_sync_scrip_master(offline_json=groww_csv, state_dir=tmp_path, broker="groww")["instruments"] == 1
    gr_lk = mb_lookup_scrip("RELIANCE", exchange="NSE", state_dir=tmp_path, broker="groww")
    assert gr_lk is not None
    assert gr_lk["token"] == "2885"

    # 4. Paper stage across multiple brokers + verify broker-native payload shapes + broker filtering on ledger
    st_z = mb_stage_or_execute_order(
        symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
        segment="EQ_INTRADAY", now_str="2026-11-09 10:15", state_dir=tmp_path, broker="zerodha",
    )
    assert st_z["broker"] == "zerodha"
    assert st_z["broker_payload"]["product"] == "MIS"
    assert st_z["broker_payload"]["tradingsymbol"] == "RELIANCE"

    st_k = mb_stage_or_execute_order(
        symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
        segment="EQ_INTRADAY", now_str="2026-11-09 10:20", state_dir=tmp_path, broker="kotak_neo",
    )
    assert st_k["broker_payload"]["es"] == "nse_cm"
    assert st_k["broker_payload"]["pc"] == "MIS"
    closed_k = mb_close_paper_trade(
        trade_id=st_k["trade_id"], exit_price=1352.0, reason="TARGET",
        now_str="2026-11-09 11:00", state_dir=tmp_path,
    )
    assert closed_k["broker"] == "kotak_neo"
    assert closed_k["total_charges_inr"] < 40.0

    st_sh = mb_stage_or_execute_order(
        symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
        segment="EQ_INTRADAY", now_str="2026-11-09 10:25", state_dir=tmp_path, broker="shoonya",
    )
    assert st_sh["broker_payload"]["prd"] == "I"
    assert st_sh["broker_payload"]["prctyp"] == "LMT"

    # Broker, symbol, and price mismatch guards when using --risk-json
    verdict_z = evaluate_risk_gate(
        capital_inr=50000.0, risk_pct=1.0, entry=1324.0, stop=1310.0, target=1352.0,
        segment="EQ_INTRADAY", symbol="RELIANCE", now_str="2026-11-11 10:15", state_dir=tmp_path, broker="zerodha",
    )
    assert verdict_z["verdict"] == "PASS"
    vpath = tmp_path / "risk_z.json"
    vpath.write_text(json.dumps(verdict_z), encoding="utf-8")
    with pytest.raises(NSETraderError, match="BROKER_MISMATCH"):
        mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", risk_json_path=str(vpath), now_str="2026-11-11 10:15", state_dir=tmp_path, broker="dhan",
        )
    with pytest.raises(NSETraderError, match="SYMBOL_MISMATCH"):
        mb_stage_or_execute_order(
            symbol="SBIN", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", risk_json_path=str(vpath), now_str="2026-11-11 10:15", state_dir=tmp_path, broker="zerodha",
        )
    with pytest.raises(NSETraderError, match="PRICE_MISMATCH"):
        mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1325.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", risk_json_path=str(vpath), now_str="2026-11-11 10:15", state_dir=tmp_path, broker="zerodha",
        )

    kotak_only = mb_read_ledger(state_dir=tmp_path, broker="kotak_neo")
    assert kotak_only["count"] == 1
    assert kotak_only["trades"][0]["broker"] == "kotak_neo"

    # 5. Mocked live HTTP dispatch verifying exact headers & payloads across ALL 9 brokers (offline)
    monkeypatch.setenv("ZERODHA_API_KEY", "zkey")
    monkeypatch.setenv("ZERODHA_ACCESS_TOKEN", "ztok")
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "utok")
    monkeypatch.setenv("DHAN_CLIENT_ID", "dcli")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "dtok")
    monkeypatch.setenv("FYERS_APP_ID", "fapp")
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "ftok")
    monkeypatch.setenv("GROWW_ACCESS_TOKEN", "gtok")
    monkeypatch.setenv("KOTAK_ACCESS_TOKEN", "ktok")
    monkeypatch.setenv("KOTAK_SID", "ksid")
    monkeypatch.setenv("ICICI_API_KEY", "ikey")
    monkeypatch.setenv("ICICI_SECRET_KEY", "isecret")
    monkeypatch.setenv("ICICI_SESSION_TOKEN", "isess")
    monkeypatch.setenv("SHOONYA_USER_ID", "suid")
    monkeypatch.setenv("SHOONYA_SESSION_TOKEN", "ssess")
    monkeypatch.setenv("ANGEL_API_KEY", "akey")
    monkeypatch.setenv("ANGEL_CLIENT_CODE", "acode")
    monkeypatch.setenv("ANGEL_MPIN", "1234")
    monkeypatch.setenv("ANGEL_TOTP_SECRET", "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ")
    captured: list[tuple[str, dict[str, str], bytes]] = []

    def _fake_urlopen(req: urllib.request.Request, timeout: int = 20) -> _DummyHTTPResp:
        _ = timeout
        raw_data = req.data if isinstance(req.data, bytes) else b""
        captured.append((req.full_url, {k.lower(): v for k, v in req.header_items()}, raw_data))
        return _DummyHTTPResp(b'{"status": true, "data": {"jwtToken": "ajwt", "order_id": "2611090001"}}')

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        live_z = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:30", state_dir=tmp_path, broker="zerodha",
        )
        assert live_z["status"] == "DISPATCHED"
        assert captured[-1][0] == "https://api.kite.trade/orders/regular"
        assert captured[-1][1]["x-kite-version"] == "3"
        assert captured[-1][1]["authorization"] == "token zkey:ztok"

        live_ic = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:32", state_dir=tmp_path, broker="icici_breeze",
        )
        assert live_ic["status"] == "DISPATCHED"
        assert captured[-1][0] == "https://api.icicidirect.com/breezeapi/api/v1/order"
        assert captured[-1][1]["x-appkey"] == "ikey"
        assert captured[-1][1]["x-sessiontoken"] == "isess"
        assert captured[-1][1]["x-checksum"].startswith("token ")
        assert len(captured[-1][1]["x-checksum"]) == 70

        # Upstox (synced above -> succeeds)
        live_up = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:34", state_dir=tmp_path, broker="upstox",
        )
        assert live_up["status"] == "DISPATCHED"
        assert captured[-1][0] == "https://api-hft.upstox.com/v3/order/place"
        assert captured[-1][1]["authorization"] == "Bearer utok"

        # Dhan, Fyers, Groww, Kotak Neo, Shoonya
        live_dh = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:35", state_dir=tmp_path, broker="dhan",
        )
        assert live_dh["status"] == "DISPATCHED"
        assert captured[-1][1]["access-token"] == "dtok"

        live_fy = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:36", state_dir=tmp_path, broker="fyers",
        )
        assert live_fy["status"] == "DISPATCHED"
        assert captured[-1][1]["authorization"] == "fapp:ftok"

        live_gr = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:37", state_dir=tmp_path, broker="groww",
        )
        assert live_gr["status"] == "DISPATCHED"
        assert captured[-1][1]["x-api-version"] == "1.0"

        live_kt = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:38", state_dir=tmp_path, broker="kotak_neo",
        )
        assert live_kt["status"] == "DISPATCHED"
        assert captured[-1][1]["sid"] == "ksid"

        live_sh = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:39", state_dir=tmp_path, broker="shoonya",
        )
        assert live_sh["status"] == "DISPATCHED"
        assert b"jKey=ssess" in captured[-1][2]

        # Angel One direct-REST fallback with symboltoken override
        live_ao = mb_stage_or_execute_order(
            symbol="RELIANCE", entry=1324.0, stop=1310.0, target=1352.0,
            segment="EQ_INTRADAY", symboltoken="2885", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-13 10:30", state_dir=tmp_path, broker="angel_one",
        )
        assert live_ao["status"] == "DISPATCHED"
        assert captured[-1][0].endswith("/placeOrder")
        assert captured[-1][1]["authorization"] == "Bearer ajwt"

    # Token-required broker without synced token must block live dispatch
    with pytest.raises(NSETraderError, match="requires symboltoken/instrument_token"):
        mb_stage_or_execute_order(
            symbol="INFY", entry=1500.0, stop=1490.0, target=1520.0,
            segment="EQ_INTRADAY", live=True, confirm_phrase="CONFIRM LIVE",
            now_str="2026-11-12 10:40", state_dir=tmp_path, broker="upstox",
        )


def test_legacy_db_migration_and_edge_cases(tmp_path: Path) -> None:
    # 1. Pre-v1.1 paper_trades.db without `broker` column migrates cleanly
    legacy_db = tmp_path / "paper_trades.db"
    conn = sqlite3.connect(str(legacy_db))
    conn.execute(
        """
        CREATE TABLE paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_ist TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            segment TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            lot_size INTEGER NOT NULL,
            planned_entry REAL NOT NULL,
            filled_entry REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_price REAL NOT NULL,
            max_risk_inr REAL NOT NULL,
            status TEXT NOT NULL,
            exit_price REAL,
            exit_time_ist TEXT,
            exit_reason TEXT,
            gross_pnl_inr REAL,
            total_charges_inr REAL,
            realized_net_pnl_inr REAL,
            risk_fingerprint TEXT,
            smartapi_payload TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO paper_orders (
            created_at_ist, trade_date, symbol, exchange, segment, side, qty, lot_size,
            planned_entry, filled_entry, stop_loss, target_price, max_risk_inr, status
        ) VALUES ('2026-11-09T10:15:00+05:30', '2026-11-09', 'RELIANCE', 'NSE', 'EQ_INTRADAY',
                  'buy', 10, 1, 1300.0, 1300.05, 1290.0, 1320.0, 110.0, 'OPEN')
        """
    )
    conn.commit()
    conn.close()

    migrated = mb_read_ledger(state_dir=tmp_path, broker="angel_one")
    assert migrated["count"] == 1
    assert migrated["trades"][0]["broker"] == "angel_one"

    # 2. format_human_report handles missing intraday/pct_change without TypeError
    sample = {
        "symbol": {"yahoo": "TEST.NS", "exchange": "NSE"},
        "session": {"ist_timestamp": "2026-11-09T10:15:00+05:30", "phase": "PRIMARY_EXECUTION", "intraday_entry_allowed": True},
        "daily": {"close": 100.0, "pdc": 100.0, "pct_change_from_pdc": None, "pdh": 102.0, "pdl": 98.0, "atr14": 2.0, "ema20": 99.0, "ema50": 95.0, "ema200": None, "rsi14": 55.0, "adx": {"adx": 25.0}, "macd": {"histogram": 0.2}, "relative_volume": 1.3},
        "weekly": {"bias": "BULLISH", "rsi14_w": 58.0},
        "intraday": {},
        "fibonacci": {},
        "rsi_divergences": [],
        "confluence": {
            "bias": "BULLISH",
            "active_confluence_score": 4,
            "score_out_of": 6,
            "reasons": ["HTF trend aligned bullish"],
            "candidate_setup": {"side": "buy", "entry": 100.0, "stop_loss": 97.0, "target_1": 105.0, "target_2": 108.0},
        },
        "regulatory": {"warnings": []},
    }
    report_txt = format_human_report(sample)
    assert "N/A" in report_txt

    # 3. Time Machine respects --broker tax drag differences (Shoonya vs ICICI delivery)
    f_1d = _make_offline_daily(tmp_path / "TCS.NS-1d.json", num_days=120, base_price=3500.0)
    tm_sh = run_timemachine(
        symbol="TCS.NS", entry=3600.0, stop=3450.0, target=3900.0,
        mode="SWING", replays=500, offline_path=str(f_1d), broker="shoonya",
    )
    tm_ic = run_timemachine(
        symbol="TCS.NS", entry=3600.0, stop=3450.0, target=3900.0,
        mode="SWING", replays=500, offline_path=str(f_1d), broker="icici_breeze",
    )
    assert tm_sh["broker"] == "shoonya"
    assert tm_ic["broker"] == "icici_breeze"
    assert tm_ic["estimated_charges_inr"] > tm_sh["estimated_charges_inr"]


