#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pathlib
import logging

import torch

<<<<<<< HEAD

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
DEFAULT_GPU = '0'
=======
<<<<<<< HEAD:scripts/extract.py
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
=======
from esm import FastaBatchedDataset, pretrained
from sys import stdout
>>>>>>> c1ff39d (added extra arg for logfile):extract.py
>>>>>>> 7e80aca (added extra arg for logfile)


DEFAULT_GPU = '0'


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file to where the logging messages will be sent"
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="The id of the GPU to use when computing the embeddings",
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args, gpu_id):
    # Set up logging
    logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging_level = logging.DEBUG
    logging_encoding = 'utf-8'
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, encoding=logging_encoding,
                            level=logging_level, format=logging_format)
    else:
        logging.basicConfig(stream=stdout, encoding=logging_encoding,
                            level=logging_level, format=logging_format)

    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    if torch.cuda.is_available() and not args.nogpu:
        model = model.to(torch.device(f'cuda:{gpu_id}'))
        logging.info("Transferred model to GPU")

        if args.gpu_id is None:
            logging.warning(f"The id for the GPU to compute the embeddings was not specified. Defaulting to {DEFAULT_GPU}")

        gpu_id = args.gpu_id if args.gpu_id is not None else DEFAULT_GPU

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    logging.info(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            logging.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device=f"cuda:{gpu_id}", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(
                    result,
                    args.output_file,
                )


def setup(args):
    # Set up logging
    logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging_level = logging.DEBUG
    logging_encoding = 'utf-8'
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, encoding=logging_encoding,
                            level=logging_level, format=logging_format)
    else:
        logging.basicConfig(stream=stdout, encoding=logging_encoding,
                            level=logging_level, format=logging_format)

    # Set up GPU
    if args.gpu_id is None:
        logging.warning(f"The id for the GPU to compute the embeddings was not specified. Defaulting to {DEFAULT_GPU}")
        gpu_id = DEFAULT_GPU
    else:
        logging.info(f"Computing embeddings on GPU {args.gpu_id}")
        gpu_id = args.gpu_id

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    return gpu_id


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    gpu_id = setup(args)
    main(args, gpu_id)
    logging.info('Successfully completed')
