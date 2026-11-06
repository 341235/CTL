import json
import math
import unittest

import pandas as pd

from src.stage3_rag.evaluation import evaluate_answers, evaluate_retrieval, judge_answer, token_f1


def qa(qa_id="q1", category="factual"):
    return {"qa_id": qa_id, "passage_id": "p1", "record_id": "r1", "question": "When was it launched?",
            "reference_answer": "The service launched in 2025.", "category": category,
            "gold_chunk_ids": ["c1", "c3"], "gold_evidence": "launched in 2025"}


def source(chunk_id, text="The service launched in 2025.", passage_id=1):
    return {"chunk_id": chunk_id, "record_id": "r1", "passage_id": passage_id,
            "text": text, "date": "2026-02-03", "title": "Report"}


def answer():
    return {"status": "ok", "answer": "The service launched in 2025.",
            "citations": [1], "supporting_quote": "launched in 2025", "abstained": False,
            "context": [source("c1")]}


class FakeJudge:
    def __init__(self, output=None, status="ok"):
        self.output = output or {"faithfulness": 5, "relevance": 4, "context_grounding": 5,
                                 "temporal_grounding": None, "explanation": "The text states the launch year."}
        self.status = status
        self.calls = []

    def generate_json(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"status": self.status, "output": self.output, "model_identity": {"name": "local-judge"}}


class RetrievalEvaluationTests(unittest.TestCase):
    def test_gold_recall_rank_and_ndcg_do_not_double_count_duplicates(self):
        runs = {"text_only": {"q1": [source("c2"), source("c3"), source("c3")]}}
        per, aggregate = evaluate_retrieval([qa()], runs, k=3)
        row = per.iloc[0]
        self.assertEqual(row.recall_at_k, 0.5)
        self.assertEqual(row.mrr, 0.5)
        self.assertAlmostEqual(row.ndcg_at_k, (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertIn("not exhaustive", row.relevance_definition)
        self.assertEqual(set(aggregate.category), {"all", "factual"})

    def test_missing_runs_are_scored_and_categories_are_preserved(self):
        per, aggregate = evaluate_retrieval(pd.DataFrame([qa(), qa("q2", "temporal")]),
                                            {"graph": {"q1": [source("c1")]}}, k=4)
        self.assertEqual(len(per), 2)
        self.assertEqual(per.loc[per.qa_id == "q2", "recall_at_k"].iloc[0], 0)
        self.assertEqual(aggregate.loc[aggregate.category == "all", "n_questions"].iloc[0], 2)

    def test_csv_gold_ids_are_accepted_and_duplicate_qa_ids_rejected(self):
        question = qa()
        question["gold_chunk_ids"] = '["c1", "c3"]'
        per, _ = evaluate_retrieval([question], {"v": {"q1": [source("c1")]}})
        self.assertEqual(per.iloc[0].recall_at_k, 0.5)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_retrieval([question, question], {"v": {}})


class AnswerEvaluationTests(unittest.TestCase):
    def test_reference_metrics_and_actual_context_grounding(self):
        result = answer()
        per, aggregate = evaluate_answers([qa()], {"text_only": {"q1": result}})
        row = per.iloc[0]
        self.assertEqual(row.token_f1, 1)
        self.assertEqual(row.exact_match, 1)
        self.assertEqual(row.citation_validity, 1)
        self.assertEqual(row.gold_context_coverage, 0.5)
        self.assertEqual(row.exact_quote_grounding_proxy, 1)
        self.assertTrue(pd.isna(row.ai_judge_faithfulness))
        self.assertTrue(pd.isna(row.temporal_reference_year_recall_proxy))
        self.assertEqual(aggregate.iloc[0].ai_judge_faithfulness_n, 0)

    def test_quote_and_citation_claims_are_recomputed_from_actual_context(self):
        result = answer()
        result.update(citations=[1, 9, True], supporting_quote="A fabricated quote", quote_valid=True,
                      citation_validity=1)
        per, _ = evaluate_answers([qa()], {"v": {"q1": result}})
        self.assertAlmostEqual(per.iloc[0].citation_validity, 1 / 3)
        self.assertEqual(per.iloc[0].exact_quote_grounding_proxy, 0)

    def test_quote_in_uncited_source_gets_no_credit(self):
        result = answer()
        result["context"] = [source("c1", "Different text"), source("c3", passage_id=2)]
        per, _ = evaluate_answers([qa()], {"v": {"q1": result}})
        self.assertEqual(per.iloc[0].exact_quote_grounding_proxy, 0)

    def test_temporal_proxies_apply_only_to_temporal_questions(self):
        result = answer()
        per, _ = evaluate_answers([qa(category="temporal"), qa("q2")],
                                  {"v": {"q1": result, "q2": result}})
        self.assertEqual(per.iloc[0].temporal_reference_year_recall_proxy, 1)
        self.assertEqual(per.iloc[0].temporal_publication_date_mentioned_proxy, 0)
        self.assertTrue(pd.isna(per.iloc[1].temporal_reference_year_recall_proxy))

    def test_missing_answers_are_not_dropped(self):
        per, aggregate = evaluate_answers([qa(), qa("q2")], {"v": {"q1": answer()}})
        self.assertEqual(per.iloc[1].status, "missing")
        self.assertEqual(per.iloc[1].token_f1, 0)
        self.assertEqual(aggregate.iloc[0].generation_success, 0.5)

    def test_abstention_without_quotes_is_not_a_fake_grounding_score(self):
        result = answer()
        result.update(answer="Insufficient evidence.", citations=[], supporting_quote=None, abstained=True)
        per, _ = evaluate_answers([qa()], {"v": {"q1": result}})
        self.assertTrue(pd.isna(per.iloc[0].citation_validity))
        self.assertTrue(pd.isna(per.iloc[0].exact_quote_grounding_proxy))
        self.assertEqual(per.iloc[0].abstention_rate, 1)

    def test_token_f1_uses_token_multiplicity_and_normalization(self):
        self.assertEqual(token_f1("The service!", "service"), 1)
        self.assertAlmostEqual(token_f1("service service", "service"), 2 / 3)

    def test_executed_ai_judge_scores_are_separate_from_lexical_proxies(self):
        result = answer()
        result["judge"] = judge_answer(qa(), result, [], FakeJudge())
        per, _ = evaluate_answers([qa()], {"v": {"q1": result}})
        self.assertEqual(per.iloc[0].ai_judge_faithfulness, 5)
        self.assertTrue(pd.isna(per.iloc[0].ai_judge_temporal_grounding))


class JudgeTests(unittest.TestCase):
    def test_judge_sees_actual_sources_and_no_reference_answer_or_gold(self):
        question = qa()
        question["reference_answer"] = "SECRET_GOLD_ANSWER"
        question["gold_evidence"] = "SECRET_GOLD_EVIDENCE"
        judge = FakeJudge()
        result = judge_answer(question, answer(), [source("unseen", "UNSEEN_SOURCE")], judge)
        messages, options = judge.calls[0]
        self.assertNotIn("SECRET_GOLD", str(messages))
        self.assertNotIn("UNSEEN_SOURCE", str(messages))
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["sources"][0]["text"], answer()["context"][0]["text"])
        self.assertEqual(result["review_method"], "ai_judge")
        self.assertEqual(options["schema"]["properties"]["temporal_grounding"], {"type": "null"})

    def test_failed_or_invalid_judges_never_receive_fabricated_scores(self):
        failed = judge_answer(qa(), answer(), [], FakeJudge(status="invalid_json"))
        self.assertIsNone(failed["faithfulness"])
        invalid = FakeJudge(output={"faithfulness": True, "relevance": 5, "context_grounding": 5,
                                    "temporal_grounding": None, "explanation": "bad type"})
        result = judge_answer(qa(), answer(), [], invalid)
        self.assertEqual(result["status"], "invalid_judge_schema")
        self.assertIsNone(result["faithfulness"])

    def test_temporal_judge_requires_a_real_temporal_score(self):
        result = judge_answer(qa(category="temporal"), answer(), [], FakeJudge())
        self.assertEqual(result["status"], "invalid_judge_schema")
        judge = FakeJudge(output={"faithfulness": 4, "relevance": 4, "context_grounding": 4,
                                  "temporal_grounding": 3, "explanation": "Publication and event dates differ."})
        self.assertEqual(judge_answer(qa(category="temporal"), answer(), [], judge)["temporal_grounding"], 3)


if __name__ == "__main__":
    unittest.main()
