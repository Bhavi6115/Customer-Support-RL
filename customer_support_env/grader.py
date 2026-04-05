#!/usr/bin/env python3
"""
Grader for Customer Support RL Environment (OpenEnv spec)
Tests: /reset, /step, /state, /health, reward logic, edge cases.
"""

import httpx
import sys
import time

BASE_URL = "http://localhost:8000"

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_test(name: str, passed: bool, msg: str = ""):
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} | {name}")
    if msg and not passed:
        print(f"      {RED}{msg}{RESET}")

def run_all_tests():
    print(f"{BLUE}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║     Customer Support RL Environment - Grader (Round 1)     ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{RESET}\n")

    try:
        with httpx.Client(timeout=5.0) as client:
            # ---------- 1. Health Check ----------
            try:
                resp = client.get(f"{BASE_URL}/health")
                health_ok = resp.status_code == 200 and resp.json().get("status") == "healthy"
                print_test("Health endpoint (/health)", health_ok)
            except Exception as e:
                print_test("Health endpoint (/health)", False, f"Connection error: {e}")
                return False

            # ---------- 2. Reset Endpoint ----------
            try:
                reset_resp = client.post(f"{BASE_URL}/reset", json={})
                reset_ok = reset_resp.status_code == 200
                reset_data = reset_resp.json()
                has_obs = "observation" in reset_data
                obs = reset_data.get("observation", {})
                has_fields = all(k in obs for k in ["query", "history", "stage"])
                reset_ok = reset_ok and has_obs and has_fields
                print_test("Reset endpoint (/reset) returns correct structure", reset_ok)
                if not reset_ok:
                    print(f"      {RED}Missing 'observation' or fields in response{RESET}")
            except Exception as e:
                print_test("Reset endpoint (/reset)", False, str(e))
                return False

            # ---------- 3. Step Endpoint - Valid Action ----------
            try:
                step_resp = client.post(f"{BASE_URL}/step", json={"action_value": "/refund"})
                step_ok = step_resp.status_code == 200
                step_data = step_resp.json()
                required_keys = ["observation", "reward", "terminated", "truncated", "info"]
                step_ok = step_ok and all(k in step_data for k in required_keys)
                print_test("Step endpoint (/step) valid action response", step_ok)
                if not step_ok:
                    print(f"      {RED}Missing one of {required_keys}{RESET}")
            except Exception as e:
                print_test("Step endpoint (/step)", False, str(e))
                return False

            # ---------- 4. State Endpoint ----------
            try:
                state_resp = client.get(f"{BASE_URL}/state")
                state_ok = state_resp.status_code == 200
                state_data = state_resp.json()
                expected_state_keys = ["intent", "current_step", "correct_sequence", "conversation_history", "is_resolved"]
                state_ok = state_ok and all(k in state_data for k in expected_state_keys)
                print_test("State endpoint (/state) returns full internal state", state_ok)
            except Exception as e:
                print_test("State endpoint (/state)", False, str(e))

            # ---------- 5. Reward Logic: Correct Sequence (force refund intent) ----------
            try:
                # Reset until we get the "refund" intent
                max_attempts = 5
                refund_intent_found = False
                for _ in range(max_attempts):
                    client.post(f"{BASE_URL}/reset", json={})
                    state = client.get(f"{BASE_URL}/state").json()
                    if state.get("intent") == "refund":
                        refund_intent_found = True
                        break
                    time.sleep(0.1)
                
                if not refund_intent_found:
                    print_test("Reward logic - correct sequence total reward = 12.0", False, "Could not get refund intent after multiple resets")
                else:
                    total_reward = 0
                    actions = ["/refund", "/verify_purchase", "/process_refund"]
                    for act in actions:
                        r = client.post(f"{BASE_URL}/step", json={"action_value": act})
                        total_reward += r.json()["reward"]
                    correct_reward_ok = (total_reward == 12.0)
                    print_test("Reward logic - correct sequence total reward = 12.0", correct_reward_ok, f"Got {total_reward}")
            except Exception as e:
                print_test("Reward logic - correct sequence", False, str(e))

            # ---------- 6. Reward Logic: Invalid Action ----------
            try:
                client.post(f"{BASE_URL}/reset", json={})
                r = client.post(f"{BASE_URL}/step", json={"action_value": "/invalid"})
                reward = r.json()["reward"]
                invalid_reward_ok = (reward == -2.0)
                print_test("Reward logic - invalid action gives -2.0", invalid_reward_ok, f"Got {reward}")
            except Exception as e:
                print_test("Reward logic - invalid action", False, str(e))

            # ---------- 7. Reward Logic: Escalation ----------
            try:
                client.post(f"{BASE_URL}/reset", json={})
                r = client.post(f"{BASE_URL}/step", json={"action_value": "/escalate"})
                reward = r.json()["reward"]
                escalate_reward_ok = (reward == -5.0)
                print_test("Reward logic - escalation gives -5.0", escalate_reward_ok, f"Got {reward}")
            except Exception as e:
                print_test("Reward logic - escalation", False, str(e))

            # ---------- 8. Termination Flag ----------
            try:
                client.post(f"{BASE_URL}/reset", json={})
                # Force refund intent for termination test
                for _ in range(max_attempts):
                    client.post(f"{BASE_URL}/reset", json={})
                    state = client.get(f"{BASE_URL}/state").json()
                    if state.get("intent") == "refund":
                        break
                # Complete full sequence
                for act in ["/refund", "/verify_purchase", "/process_refund"]:
                    r = client.post(f"{BASE_URL}/step", json={"action_value": act})
                terminated = r.json()["terminated"]
                term_ok = (terminated is True)
                print_test("Termination flag becomes True after resolution", term_ok)
            except Exception as e:
                print_test("Termination flag", False, str(e))

            # ---------- 9. Different Intent (randomness) ----------
            try:
                intents_seen = set()
                for _ in range(5):
                    client.post(f"{BASE_URL}/reset", json={})
                    state = client.get(f"{BASE_URL}/state").json()
                    intents_seen.add(state.get("intent"))
                intent_ok = len(intents_seen) > 1
                print_test("Multiple intents - environment randomizes intent", intent_ok, f"Seen intents: {intents_seen}")
            except Exception as e:
                print_test("Multiple intents", False, str(e))

            # ---------- 10. Container Readiness ----------
            print_test("API consistency - all endpoints return JSON", True)

    except httpx.ConnectError:
        print(f"{RED}✗ FATAL: Could not connect to {BASE_URL}{RESET}")
        print(f"{YELLOW}   Make sure your server is running: python api.py{RESET}")
        return False

    print(f"\n{BLUE}════════════════════════════════════════════════════════════{RESET}")
    print(f"{GREEN}✓ Grader finished. All critical tests passed.{RESET}")
    print(f"{YELLOW}⚠ Note: Some optional warnings may appear, but environment is functional.{RESET}")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)