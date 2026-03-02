import unittest
from fractions import Fraction

from benchmark_prism_gsm8k import (
    ConversationSampler,
    IntersectionalSampler,
    PrismConversation,
    build_prompt,
    compute_group_metrics,
    extract_gold_answer,
    extract_number_fraction,
    get_demographic_value,
    parse_intersectional_buckets,
    race_from_ethnicity_value,
    select_examples,
)


def make_conv(conversation_id: str, race: str, gender: str) -> PrismConversation:
    return PrismConversation(
        conversation_id=conversation_id,
        user_id=conversation_id,
        history_text=f"history-{conversation_id}",
        demographics={"race": race, "gender": gender},
    )


class TestBenchmarkPrismGsm8k(unittest.TestCase):
    def test_extract_gold_answer_from_gsm8k_format(self) -> None:
        self.assertEqual(extract_gold_answer("reasoning lines\n#### 42"), Fraction(42, 1))
        self.assertEqual(extract_gold_answer("reasoning lines\n#### -3/2"), Fraction(-3, 2))

    def test_extract_number_fraction_prefers_boxed_value(self) -> None:
        output = "scratch 9 then final \\boxed{13/2}"
        self.assertEqual(extract_number_fraction(output), Fraction(13, 2))

    def test_build_prompt_contains_deepseek_directives(self) -> None:
        prompt = build_prompt("What is 2+2?", make_conv("c1", "white", "male"))
        self.assertIn("Please reason step by step, and put your final answer within \\boxed{}.", prompt)
        self.assertIn("Return only your final response", prompt)
        self.assertIn("Next user message:", prompt)
        self.assertNotIn("Math question:", prompt)

    def test_parse_intersectional_buckets_parses_and_validates(self) -> None:
        self.assertEqual(
            parse_intersectional_buckets("white|male,black|female", 2),
            [("white", "male"), ("black", "female")],
        )
        with self.assertRaises(ValueError):
            parse_intersectional_buckets("white|male|extra", 2)

    def test_conversation_sampler_without_replacement_visits_all_before_repeat(self) -> None:
        items = [make_conv("a", "white", "male"), make_conv("b", "black", "female"), make_conv("c", "white", "female")]
        sampler = ConversationSampler(items, strategy="without_replacement", seed=123)
        first_epoch = {sampler.sample().conversation_id for _ in range(3)}
        self.assertEqual(first_epoch, {"a", "b", "c"})

    def test_intersectional_sampler_balances_available_buckets(self) -> None:
        items = [
            make_conv("wm1", "white", "male"),
            make_conv("wm2", "white", "male"),
            make_conv("wf1", "white", "female"),
            make_conv("bf1", "black", "female"),
        ]
        sampler = IntersectionalSampler(
            items=items,
            fields=["race", "gender"],
            target_buckets=[("white", "male"), ("white", "female"), ("black", "male"), ("black", "female")],
            strategy="without_replacement",
            seed=7,
        )

        seen_buckets = [sampler.sample()[1] for _ in range(9)]
        self.assertEqual(set(seen_buckets), {"white|male", "white|female", "black|female"})
        counts = {bucket: seen_buckets.count(bucket) for bucket in set(seen_buckets)}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_compute_group_metrics_accuracy(self) -> None:
        rows = [
            {"gender": "male", "correct": 1},
            {"gender": "male", "correct": 0},
            {"gender": "female", "correct": 1},
            {"gender": "female", "correct": 1},
        ]
        metrics = compute_group_metrics(rows, ["gender"])
        self.assertEqual(metrics["gender"]["male"]["accuracy"], 0.5)
        self.assertEqual(metrics["gender"]["female"]["accuracy"], 1.0)

    def test_select_examples_first_n(self) -> None:
        data = [{"id": i} for i in range(10)]
        selected = select_examples(data, num_questions=3, sampling="first_n", seed=123)
        self.assertEqual([row["id"] for row in selected], [0, 1, 2])

    def test_select_examples_random_is_seeded(self) -> None:
        data = [{"id": i} for i in range(10)]
        a = select_examples(data, num_questions=4, sampling="random", seed=99)
        b = select_examples(data, num_questions=4, sampling="random", seed=99)
        self.assertEqual([row["id"] for row in a], [row["id"] for row in b])
        self.assertNotEqual([row["id"] for row in a], [0, 1, 2, 3])

    def test_race_from_ethnicity_json(self) -> None:
        value = '{"self_described":"african/black","categorised":"Black / African","simplified":"Black"}'
        self.assertEqual(race_from_ethnicity_value(value), "Black")

    def test_get_demographic_value_race_from_ethnicity_when_race_missing(self) -> None:
        demographics = {
            "ethnicity": '{"self_described":"caucasian","categorised":"White","simplified":"White"}',
            "gender": "male",
        }
        self.assertEqual(get_demographic_value(demographics, "race"), "White")

if __name__ == "__main__":
    unittest.main()
