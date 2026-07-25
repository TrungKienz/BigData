import unittest

from fraud_pipeline import PipelineConfig, derive_account_state_updates, parse_csv_row


def sample_row(**overrides: str) -> dict[str, str]:
    row = {
        "step": "1",
        "type": "TRANSFER",
        "amount": "181.0",
        "nameOrig": "C1305486145",
        "oldbalanceOrg": "181.0",
        "newbalanceOrig": "0.0",
        "nameDest": "C553264065",
        "oldbalanceDest": "0.0",
        "newbalanceDest": "0.0",
        "isFraud": "1",
    }
    row.update(overrides)
    return row


class ParsingTests(unittest.TestCase):
    def test_parse_csv_row_builds_deterministic_event(self) -> None:
        config = PipelineConfig(step_seconds=60)
        event = parse_csv_row(sample_row(), config=config)

        self.assertEqual(event.step, 1)
        self.assertEqual(event.txn_type, "TRANSFER")
        self.assertEqual(event.amount, 181.0)
        self.assertTrue(event.event_id)
        self.assertTrue(event.event_time.isoformat().startswith("2026-01-01T00:01:00"))

    def test_parse_csv_row_event_id_ignores_post_transaction_balances(self) -> None:
        first = parse_csv_row(sample_row())
        second = parse_csv_row(sample_row(newbalanceDest="5.0", newbalanceOrig="5.0"))

        self.assertEqual(first.event_id, second.event_id)

    def test_parse_csv_row_event_id_changes_when_inference_balance_changes(self) -> None:
        first = parse_csv_row(sample_row())
        second = parse_csv_row(sample_row(oldbalanceDest="5.0"))

        self.assertNotEqual(first.event_id, second.event_id)

    def test_event_id_does_not_depend_on_fraud_label(self) -> None:
        legitimate = parse_csv_row(sample_row(isFraud="0"))
        fraudulent = parse_csv_row(sample_row(isFraud="1"))

        self.assertEqual(legitimate.event_id, fraudulent.event_id)

    def test_derive_account_state_updates_returns_sender_and_receiver(self) -> None:
        event = parse_csv_row(sample_row())
        updates = derive_account_state_updates(event)

        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0].account_id, "C1305486145")
        self.assertEqual(updates[0].role, "sender")
        self.assertEqual(updates[1].account_id, "C553264065")
        self.assertEqual(updates[1].role, "receiver")


if __name__ == "__main__":
    unittest.main()
