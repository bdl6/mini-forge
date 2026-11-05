from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mini_forge.data import prepare_movielens
from mini_forge.generator import SIDDataset, TokenTrie, build_vocabulary
from mini_forge.opq import encode_opq, fit_opq
from mini_forge.rqvae import RQVAE, ResidualQuantizer, load_rqvae, train_rqvae
from mini_forge.utils import read_jsonl, write_jsonl


class CoreTests(unittest.TestCase):
    def test_opq_codes_reconstruct_and_are_reproducible(self) -> None:
        generator = torch.Generator().manual_seed(11)
        vectors = torch.randn(40, 8, generator=generator)
        original = vectors.clone()
        first = fit_opq(vectors, 2, 4, rotation_iterations=2, kmeans_iterations=3, seed=7)
        second = fit_opq(vectors, 2, 4, rotation_iterations=2, kmeans_iterations=3, seed=7)
        codes, reconstruction = encode_opq(vectors, first.rotation, first.codebooks)
        torch.testing.assert_close(vectors, original)
        torch.testing.assert_close(first.rotation.T @ first.rotation, torch.eye(8), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(codes, first.codes)
        torch.testing.assert_close(reconstruction, first.reconstruction)
        torch.testing.assert_close(first.codes, second.codes)
        self.assertTrue(torch.isfinite(reconstruction).all())

    def test_rqvae_shapes_and_gradients(self) -> None:
        model = RQVAE(16, 12, 4, 3, 8)
        values = torch.randn(10, 16)
        output = model(values)
        self.assertEqual(output["reconstruction"].shape, values.shape)
        self.assertEqual(output["codes"].shape, (10, 3))
        loss = torch.nn.functional.mse_loss(output["reconstruction"], values) + output["vq_loss"]
        loss.backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_ema_codebook_update_and_eval_freeze(self) -> None:
        quantizer = ResidualQuantizer(1, 2, 1, codebook_update="ema", ema_decay=0.5)
        with torch.no_grad():
            quantizer.codebooks[0].weight.copy_(torch.tensor([[0.0], [10.0]]))
        quantizer.reset_ema_stats()
        values = torch.tensor([[2.0], [9.0]], requires_grad=True)
        quantized, commitment, codes = quantizer(values)
        self.assertEqual(codes.tolist(), [[0], [1]])
        torch.testing.assert_close(quantizer.codebooks[0].weight, torch.tensor([[1.0], [9.5]]))
        (quantized.sum() + commitment).backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNone(quantizer.codebooks[0].weight.grad)

        old_weights = quantizer.codebooks[0].weight.detach().clone()
        quantizer(values.detach(), update_ema=False)
        torch.testing.assert_close(quantizer.codebooks[0].weight, old_weights)
        quantizer.eval()
        quantizer(values.detach())
        torch.testing.assert_close(quantizer.codebooks[0].weight, old_weights)

        restored = ResidualQuantizer(1, 2, 1, codebook_update="ema", ema_decay=0.5)
        restored.load_state_dict(quantizer.state_dict())
        torch.testing.assert_close(restored.codebooks[0].weight, old_weights)
        torch.testing.assert_close(restored.ema_counts, quantizer.ema_counts)

    def test_ema_training_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content_dir = root / "artifacts" / "content"
            processed_dir = root / "processed"
            content_dir.mkdir(parents=True)
            processed_dir.mkdir()
            np.save(content_dir / "movie_embeddings.npy", np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32))
            np.save(content_dir / "movie_ids.npy", np.arange(1, 13, dtype=np.int64))
            np.save(processed_dir / "i2i_pairs.npy", np.array([[1, 2, 2.0], [3, 4, 1.0]]))
            config = {
                "seed": 7, "device": "cpu",
                "paths": {"artifact_dir": str(root / "artifacts"), "processed_dir": str(processed_dir)},
                "rqvae": {
                    "input_dim": 4, "hidden_dim": 8, "latent_dim": 2,
                    "num_codebooks": 2, "codebook_size": 4, "commitment_beta": 0.25,
                    "codebook_update": "ema", "ema_decay": 0.5, "ema_eps": 1e-5,
                    "kmeans_iterations": 2, "validation_ratio": 0.25,
                    "batch_size": 4, "pair_batch_size": 2, "i2i_weight": 0.1,
                    "temperature": 0.1, "learning_rate": 0.001, "weight_decay": 0.0,
                    "epochs": 1, "warmup_epochs": 0, "gradient_clip": 1.0,
                    "early_stop_patience": 1,
                },
            }
            summary = train_rqvae(config)
            self.assertEqual(summary["epochs_trained"], 1)
            loaded = load_rqvae(root / "artifacts" / "rqvae" / "best.pt", torch.device("cpu"))
            self.assertEqual(loaded.model_config["codebook_update"], "ema")
            self.assertIsNotNone(loaded.quantizer.ema_counts)
            self.assertTrue(torch.isfinite(loaded(torch.randn(2, 4))["reconstruction"]).all())

    def test_vocabulary_and_trie(self) -> None:
        mapping = {
            "1": {"tokens": ["<L1_0>", "<L2_1>", "<C_0>"]},
            "2": {"tokens": ["<L1_0>", "<L2_2>", "<C_0>"]},
        }
        vocabulary = build_vocabulary(mapping)
        sequences = [
            [vocabulary[token] for token in value["tokens"]]
            for value in mapping.values()
        ]
        trie = TokenTrie(sequences)
        self.assertEqual(trie.allowed([]), [vocabulary["<L1_0>"]])
        self.assertEqual(
            set(trie.allowed([vocabulary["<L1_0>"]])),
            {vocabulary["<L2_1>"], vocabulary["<L2_2>"]},
        )

    def test_structured_user_input(self) -> None:
        mapping = {
            "1": {"tokens": ["<L1_4>", "<L2_2>", "<C_0>"]},
            "2": {"tokens": ["<L1_5>", "<L2_3>", "<C_0>"]},
        }
        example = {
            "user_id": 11, "gender": "M", "age": "18", "occupation": "45",
            "history": [1, 2], "target": 2,
        }
        vocabulary = build_vocabulary(mapping, [example])
        self.assertEqual(vocabulary["<PAD>"], 0)
        self.assertEqual(vocabulary["<EOS>"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            write_jsonl(path, [example])
            row = SIDDataset(path, mapping, vocabulary)[0]
        reverse = {index: token for token, index in vocabulary.items()}
        self.assertEqual([reverse[index] for index in row["input_ids"]], [
            "<USER>", "<USER_ID_11>", "<USER_GENDER_M>", "<USER_AGE_18>",
            "<USER_OCCUPATION_45>", "</USER>", "<ITEM_LIST>",
            *mapping["1"]["tokens"], *mapping["2"]["tokens"], "</ITEM_LIST>",
        ])
        self.assertEqual([reverse[index] for index in row["labels"]], [
            *mapping["2"]["tokens"], "<EOS>",
        ])

    def test_old_examples_require_preparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "preparation"):
            build_vocabulary({}, [{"user_id": 1}])

    def test_prepare_leave_one_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "movies.dat").write_text(
                "1::A::Drama\n2::B::Comedy\n3::C::Action\n4::D::Drama\n5::E::Comedy\n",
                encoding="latin-1",
            )
            ratings = []
            for user in [1, 2]:
                for timestamp, movie in enumerate([1, 2, 3, 4, 5], start=1):
                    ratings.append(f"{user}::{movie}::5::{timestamp}")
            (raw / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
            config = {
                "paths": {"raw_dir": str(raw), "processed_dir": str(root / "processed")},
                "data": {
                    "positive_rating": 4,
                    "min_user_interactions": 3,
                    "min_item_interactions": 1,
                    "min_history": 2,
                    "max_history": 3,
                    "stride": 1,
                    "i2i_window": 2,
                },
            }
            metadata = prepare_movielens(config)
            self.assertEqual(metadata["num_users"], 2)
            self.assertEqual(metadata["num_validation_examples"], 2)
            self.assertEqual(metadata["num_test_examples"], 2)
            self.assertEqual(metadata["num_train_examples"], 2)
            train_path = root / "processed" / "train.jsonl"
            fallback = read_jsonl(train_path)[0]
            self.assertEqual(fallback["gender"], "UNK")
            (raw / "users.dat").write_text(
                "1::F::18::10::00000\n2::M::25::15::00000\n", encoding="latin-1",
            )
            prepare_movielens(config)
            profile = read_jsonl(train_path)[0]
            self.assertEqual(profile["user_id"], 1)
            self.assertEqual(profile["gender"], "F")
            self.assertEqual(profile["age"], "18")
            self.assertEqual(profile["occupation"], "10")
            self.assertTrue((root / "processed" / "users.csv").exists())


if __name__ == "__main__":
    unittest.main()
