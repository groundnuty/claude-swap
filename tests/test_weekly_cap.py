"""Per-account weekly cap (`cswap cap`): a hard limit on one account's weekly
quota. At or over it the account is never a switch target and an active one is
left at once; it rejoins rotation when its weekly window resets below the cap.

Unlike a per-account threshold, which is a switch-away line the engine relaxes
when every account is over its line, a cap is never relaxed. Each engine test
that depends on that pairs the capped run with the same fleet uncapped, so it
proves the cap changed the outcome.

Synthetic fixtures only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_swap import cli
from claude_swap.autoswitch import NoSwitchEvent, TickOutcome, WeeklyCapEvent
from claude_swap.exceptions import ConfigError
from claude_swap.json_output import account_row
from claude_swap.lines import over_weekly_cap
from claude_swap.models import AccountPolicy, normalize_weekly_cap
from claude_swap.switcher import ClaudeAccountSwitcher, _weekly_cap_marker
from tests.test_account_policy_cli import PolicyCliBase
from tests.test_autoswitch import EngineHarness, _entry_for, _usage7

SOON = "2026-11-24T00:00:00Z"
LATER = "2026-11-28T00:00:00Z"


def _fleet(temp_home: Path, n: int = 3, **settings) -> EngineHarness:
    """Accounts 1..n, account 1 live and active. Production settings unless
    overridden: consume-first at a 98% line."""
    settings.setdefault("strategy", "consume-first")
    settings.setdefault("threshold", 98.0)
    h = EngineHarness(temp_home, **settings)
    for i in range(1, n + 1):
        h.seed(i, f"acct{i}@example.com")
    h.make_live("acct1@example.com", 1)
    return h


def _cap(h: EngineHarness, num: int, pct: float) -> None:
    h.switcher.set_account_weekly_cap(str(num), pct)
    h.engine = h._make_engine()


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


def _mark_api_key(h: EngineHarness, num: int) -> None:
    data = h.switcher._get_sequence_data()
    data["accounts"][str(num)]["kind"] = "api_key"
    h.switcher._write_json(h.switcher.sequence_file, data)


class TestNeverALanding:
    # 1 is exhausted. 2 is over its weekly line with a sliver of room; 3 has
    # far more room but is past an 80% cap. The all-over fallback lands on the
    # most room, which is 3, unless the cap takes 3 out.
    USAGE = {
        "1": _usage7(100, 60, LATER),
        "2": _usage7(40, 99, LATER),
        "3": _usage7(10, 81, SOON),
    }

    @pytest.mark.parametrize("strategy", ["consume-first", "best", "weekly-first"])
    def test_without_the_cap_the_fallback_takes_the_most_room(self, temp_home, strategy):
        h = _fleet(temp_home, strategy=strategy)
        assert h.tick_with_usage(dict(self.USAGE)) is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    @pytest.mark.parametrize("strategy", ["consume-first", "best", "weekly-first"])
    def test_a_capped_account_is_not_landed_on_even_when_all_are_over(
        self, temp_home, strategy
    ):
        h = _fleet(temp_home, strategy=strategy)
        _cap(h, 3, 80)
        assert h.tick_with_usage(dict(self.USAGE)) is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2

    def test_when_every_other_account_is_capped_it_says_so(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        outcome = h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 85)})
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert "weekly-cap" in _reasons(h)

    def test_below_its_cap_the_account_is_used_as_usual(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        assert h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 50)}) \
            is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_after_its_weekly_reset_the_account_is_back(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 85)})
        assert h.active_number() == 1
        h.clock.advance(600)
        assert h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 3)}) \
            is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_clearing_the_cap_returns_the_account_at_once(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 85)})
        h.switcher.set_account_weekly_cap("2", None)
        h.engine = h._make_engine()
        h.clock.advance(600)
        assert h.tick_with_usage({"1": _usage7(100, 60), "2": _usage7(0, 85)}) \
            is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_an_unread_weekly_window_is_not_trusted(self, temp_home):
        # 2 reports only its 5h window. Uncapped it is the most room; capped,
        # nothing shows it is under the cap, so 3 is taken.
        usage = {
            "1": _usage7(100, 60),
            "2": {"five_hour": {"pct": 5.0}},
            "3": _usage7(50, 60),
        }
        h = _fleet(temp_home)
        _cap(h, 2, 80)
        assert h.tick_with_usage(dict(usage)) is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def test_a_higher_priority_tier_is_skipped_while_capped(self, temp_home):
        h = _fleet(temp_home, strategy="priority-consume-first")
        h.switcher.set_account_priority("2", 100)
        _cap(h, 2, 80)
        outcome = h.tick_with_usage({
            "1": _usage7(100, 60), "2": _usage7(0, 85), "3": _usage7(0, 10),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3


class TestLeavingAtTheCap:
    def test_the_active_account_leaves_at_its_cap_below_every_line(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 1, 80)
        outcome = h.tick_with_usage({"1": _usage7(10, 80), "2": _usage7(10, 20)})
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2
        caps = [e for e in h.events if isinstance(e, WeeklyCapEvent)]
        assert [(e.account["number"], e.pct, e.cap) for e in caps] == [(1, 80.0, 80.0)]
        assert "weekly cap" in caps[0].human()

    def test_below_its_cap_the_active_account_stays(self, temp_home):
        h = _fleet(temp_home, n=2, strategy="best")
        _cap(h, 1, 80)
        assert h.tick_with_usage({"1": _usage7(10, 79), "2": _usage7(10, 20)}) \
            is TickOutcome.NO_ACTION
        assert h.active_number() == 1

    def test_the_cap_does_not_wait_out_the_cooldown(self, temp_home):
        h = _fleet(temp_home)
        h.tick_with_usage({
            "1": _usage7(100, 60), "2": _usage7(10, 20), "3": _usage7(10, 20),
        })
        landed = h.active_number()
        assert landed in (2, 3)
        other = 5 - landed
        _cap(h, landed, 80)
        h.clock.advance(60)   # well inside the 5-minute cooldown
        outcome = h.tick_with_usage({
            "1": _usage7(100, 60), str(landed): _usage7(10, 80), str(other): _usage7(10, 20),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == other

    def test_it_leaves_even_for_less_room_than_it_has(self, temp_home):
        # 1 is past its cap with 19 points of raw room; 2 is over its line
        # with 1. Measured against staying, the all-over fallback would hold;
        # a capped account's own room counts as none, so it leaves.
        h = _fleet(temp_home, n=2)
        _cap(h, 1, 80)
        assert h.tick_with_usage({"1": _usage7(10, 81), "2": _usage7(30, 99)}) \
            is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2

    def test_under_priority_a_capped_top_tier_leaves(self, temp_home):
        h = _fleet(temp_home, n=2, strategy="priority-consume-first")
        h.switcher.set_account_priority("1", 100)
        _cap(h, 1, 80)
        assert h.tick_with_usage({"1": _usage7(10, 85), "2": _usage7(10, 20)}) \
            is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2

    def test_a_candidate_that_crosses_its_cap_on_the_fresh_reading_is_dropped(
        self, temp_home
    ):
        # consume-first re-reads usage just before a below-the-line move and
        # re-ranks on it. 2 resets sooner and is under its cap on the first
        # reading (70) but at it on the fresh one (80).
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        first = {"1": _usage7(10, 50, LATER), "2": _usage7(10, 70, SOON)}
        fresh = {"1": _usage7(10, 50, LATER), "2": _usage7(10, 80, SOON)}
        fresh_taken = []

        def entries(*_a, fetch=frozenset(), **_k):
            # The pre-switch re-read is the one that fetches the active too.
            fresh_taken.append("1" in (fetch or ()))
            use = fresh if fresh_taken[-1] else first
            return {n: _entry_for(v, h.clock()) for n, v in use.items()}

        from unittest.mock import patch
        with patch.object(h.switcher, "usage_entries_by_account", side_effect=entries):
            outcome = h.engine.tick()
        assert any(fresh_taken), "no fresh reading was taken"
        assert outcome is not TickOutcome.SWITCHED and h.active_number() == 1

    def test_the_same_move_without_the_crossing_happens(self, temp_home):
        h = _fleet(temp_home, n=2)
        _cap(h, 2, 80)
        assert h.tick_with_usage({"1": _usage7(10, 50, LATER), "2": _usage7(10, 70, SOON)}) \
            is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_an_unread_weekly_window_does_not_force_a_departure(self, temp_home):
        h = _fleet(temp_home, n=2, strategy="best")
        _cap(h, 1, 80)
        assert h.tick_with_usage({"1": {"five_hour": {"pct": 10.0}}, "2": _usage7(10, 20)}) \
            is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        assert not any(isinstance(e, WeeklyCapEvent) for e in h.events)


class TestWithAnApiKey:
    def test_a_capped_active_goes_to_the_key_when_the_rest_is_spent(self, temp_home):
        h = _fleet(temp_home, include_api_key_accounts=True, threshold=95.0)
        _mark_api_key(h, 3)
        _cap(h, 1, 80)
        outcome = h.tick_with_usage({
            "1": _usage7(10, 85), "2": _usage7(99, 99), "3": "api key",
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def test_under_priority_a_capped_top_tier_goes_to_the_key(self, temp_home):
        h = _fleet(temp_home, strategy="priority-consume-first",
                   include_api_key_accounts=True, threshold=95.0)
        _mark_api_key(h, 3)
        h.switcher.set_account_priority("1", 100)
        _cap(h, 1, 80)
        outcome = h.tick_with_usage({
            "1": _usage7(10, 85), "2": _usage7(99, 99), "3": "api key",
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def _on_the_key(self, temp_home, cap: bool) -> EngineHarness:
        h = EngineHarness(temp_home, include_api_key_accounts=True, threshold=95.0,
                          strategy="consume-first")
        h.seed(1, "key@token.local")
        h.seed(2, "capped@example.com")
        h.make_live("key@token.local", 1)
        _mark_api_key(h, 1)
        if cap:
            _cap(h, 2, 80)
        # 82: under the 95 line by more than the 10-point return margin, and
        # over an 80 cap.
        h.tick_with_usage({"1": "api key", "2": _usage7(0, 82)})
        return h

    def test_leaving_the_key_returns_to_a_subscription_without_a_cap(self, temp_home):
        assert self._on_the_key(temp_home, cap=False).active_number() == 2

    def test_leaving_the_key_skips_a_capped_subscription(self, temp_home):
        h = self._on_the_key(temp_home, cap=True)
        assert h.active_number() == 1
        assert _reasons(h) == ["active-api-key"]


class TestStore(PolicyCliBase):
    def test_set_and_clear(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        s.set_account_weekly_cap("2", "80")
        assert self._record(ClaudeAccountSwitcher(), "2")["weeklyCap"] == 80.0
        assert s.account_policies()["2"].weekly_cap == 80.0
        s.set_account_weekly_cap("2", None)
        assert "weeklyCap" not in self._record(ClaudeAccountSwitcher(), "2")

    @pytest.mark.parametrize("bad", ["0", "100", "150", "abc", "nan", True])
    def test_a_bad_value_writes_nothing(self, temp_home: Path, bad):
        s = self._fleet(temp_home)
        before = self._bytes(s)
        with pytest.raises(ValueError, match="weekly cap must be between 1 and 99.9"):
            s.set_account_weekly_cap("2", bad)
        assert self._bytes(s) == before

    def test_an_api_key_account_cannot_be_capped(self, temp_home: Path):
        s = self._fleet(temp_home)
        data = s._get_sequence_data()
        data["accounts"]["2"]["kind"] = "api_key"
        s._write_json(s.sequence_file, data)
        with pytest.raises(ConfigError, match="no weekly quota"):
            s.set_account_weekly_cap("2", 80)

    def test_a_hand_edited_cap_degrades_to_none(self, temp_home: Path):
        s = self._fleet(temp_home)
        data = s._get_sequence_data()
        data["accounts"]["2"]["weeklyCap"] = "lots"
        assert ClaudeAccountSwitcher._policy_from_data(data, "2").weekly_cap is None

    def test_the_cap_survives_a_slot_re_add(self):
        assert ClaudeAccountSwitcher._carried_settings(
            {"email": "x", "weeklyCap": 80.0}
        ) == {"weeklyCap": 80.0}


class TestCli(PolicyCliBase):
    @staticmethod
    def _cap(argv: list[str]) -> None:
        from unittest.mock import patch
        with patch("os.geteuid", return_value=1000, create=True):
            cli._cap_command(argv)

    def test_set_list_and_unset(self, temp_home: Path, capsys):
        self._fleet(temp_home)
        self._cap(["2", "80"])
        assert self._record(ClaudeAccountSwitcher(), "2")["weeklyCap"] == 80.0
        self._cap([])
        assert "Account-2  80% of its weekly quota" in capsys.readouterr().out
        self._cap(["2", "--unset"])
        assert "weeklyCap" not in self._record(ClaudeAccountSwitcher(), "2")

    def test_reachable_through_main(self, temp_home: Path):
        self._fleet(temp_home)
        self._main(["cap", "2", "80"])
        assert self._record(ClaudeAccountSwitcher(), "2")["weeklyCap"] == 80.0

    def test_a_bad_value_exits_nonzero(self, temp_home: Path):
        s = self._fleet(temp_home)
        before = self._bytes(s)
        with pytest.raises(SystemExit) as exc:
            self._cap(["2", "150"])
        assert exc.value.code != 0 and self._bytes(s) == before


class TestDisplay:
    @staticmethod
    def _row(usage, cap):
        return account_row(
            2, "b@example.com", "", "", False, usage,
            policy=AccountPolicy(weekly_cap=cap),
        )

    def test_list_json_reports_the_cap_and_whether_it_is_reached(self):
        assert self._row(_usage7(10, 85), 80.0)["weeklyCap"] == 80.0
        assert self._row(_usage7(10, 85), 80.0)["capped"] is True
        assert self._row(_usage7(10, 50), 80.0)["capped"] is False
        assert "capped" not in self._row(None, 80.0)
        assert "weeklyCap" not in self._row(_usage7(10, 85), None)

    def test_over_weekly_cap(self):
        assert over_weekly_cap([("5h", 10.0, None), ("7d", 80.0, None)], 80.0, unknown=False)
        assert not over_weekly_cap([("7d", 79.9, None)], 80.0, unknown=True)
        assert over_weekly_cap([("5h", 10.0, None)], 80.0, unknown=True)
        assert not over_weekly_cap([("5h", 10.0, None)], 80.0, unknown=False)
        assert not over_weekly_cap([("7d", 99.0, None)], None, unknown=True)

    def test_normalize(self):
        assert normalize_weekly_cap("80") == 80.0
        assert normalize_weekly_cap(1) == 1.0


def test_the_list_marker_reads_the_list_s_own_entries():
    # `cswap list` hands the marker UsageEntry objects, not usage dicts.
    entry = _entry_for(_usage7(10, 81), 0.0)
    assert "(capped: 7d 81% >= 80%)" in _weekly_cap_marker(80.0, entry)
    assert "(cap 80%)" in _weekly_cap_marker(80.0, _entry_for(_usage7(10, 50), 0.0))
    assert "(cap 80%)" in _weekly_cap_marker(80.0, _entry_for("token expired", 0.0))


def test_the_list_marker():
    assert _weekly_cap_marker(None, _usage7(10, 90)) == ""
    assert "(cap 80%)" in _weekly_cap_marker(80.0, _usage7(10, 50))
    assert "(capped: 7d 81% >= 80%)" in _weekly_cap_marker(80.0, _usage7(10, 81))
    assert "(cap 80%)" in _weekly_cap_marker(80.0, "token expired")
    assert "(cap 80%)" in _weekly_cap_marker(80.0, None)
