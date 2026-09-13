"""Testes para o módulo de transliteração e normalização sânscrita."""

from __future__ import annotations

import unittest

from vedic_pipeline.common.sanskrit import (
    devanagari_to_iast,
    fold_for_search,
    get_sanskrit_variants,
    has_devanagari,
    iast_to_ascii,
)


class SanskritTests(unittest.TestCase):
    def test_has_devanagari(self):
        self.assertTrue(has_devanagari("आत्मन्"))
        self.assertTrue(has_devanagari("rigveda: ॐ"))
        self.assertFalse(has_devanagari("ātman"))
        self.assertFalse(has_devanagari("atman"))

    def test_devanagari_to_iast(self):
        # Pranava
        self.assertEqual(devanagari_to_iast("ॐ"), "om")
        # Atman
        self.assertEqual(devanagari_to_iast("आत्मन्"), "ātman")
        # Yoga
        self.assertEqual(devanagari_to_iast("योग"), "yoga")
        # Veda
        self.assertEqual(devanagari_to_iast("वेद"), "veda")
        # Karma
        self.assertEqual(devanagari_to_iast("कर्म"), "karma")
        # Empty string
        self.assertEqual(devanagari_to_iast(""), "")

    def test_fold_strips_vedic_accents(self):
        self.assertEqual(fold_for_search("इ॒षे त्वो॒र्जे"), fold_for_search("इषे त्वोर्जे"))
        self.assertEqual(fold_for_search("agním īḷe"), fold_for_search("agnim ile"))
        self.assertIn(
            fold_for_search("कर्मण्येवाधिकारस्ते"),
            fold_for_search("2.47 कर्मण्येवाधिकारस्ते मा फलेषु"),
        )

    def test_iast_to_ascii(self):
        self.assertEqual(iast_to_ascii("ātman"), "atman")
        self.assertEqual(iast_to_ascii("kṛṣṇa"), "krsna")
        self.assertEqual(iast_to_ascii("ṛgveda"), "rgveda")
        self.assertEqual(iast_to_ascii("īśvara"), "isvara")
        self.assertEqual(iast_to_ascii("Upaniṣad"), "Upanisad")

    def test_get_sanskrit_variants(self):
        # Devanagari gera IAST e ASCII
        v_deva = get_sanskrit_variants("आत्मन्")
        self.assertIn("आत्मन्", v_deva)
        self.assertIn("ātman", v_deva)
        self.assertIn("atman", v_deva)

        # IAST gera ASCII e variantes fonéticas
        v_iast = get_sanskrit_variants("kṛṣṇa")
        self.assertIn("kṛṣṇa", v_iast)
        self.assertIn("krsna", v_iast)
        self.assertIn("krishna", v_iast)


if __name__ == "__main__":
    unittest.main()
