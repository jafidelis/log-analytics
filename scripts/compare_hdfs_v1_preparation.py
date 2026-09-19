import json
from collections import defaultdict
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Callable

import jsonpickle
from drain3 import TemplateMiner

from log_analytics.data.fit_hdfs_v1 import build_demo_parser
from log_analytics.data.prepare_hdfs_v1 import (
    # prepare_hdfs_message,
    prepare_hdfs_message_candidate,
    prepare_hdfs_message_candidate_c,
)
from log_analytics.data.session_hdfs_v1 import (
    SessionLogEvent,
    associate_log_line,
    load_split_index,
)


TARGET_EVENTS = 5_000
MAX_LINES = 100_000
PrepareFunction = Callable[[str], str]


def select_sample() -> list[SessionLogEvent]:
    split_index = load_split_index(
        Path("artifacts/data_splits/hdfs_v1_session_split.csv")
    )
    sample = []

    with Path("data/raw/hdfs_v1/HDFS.log").open(
        encoding="utf-8"
    ) as stream:
        for line_id, line in enumerate(
            islice(stream, MAX_LINES),
            start=1,
        ):
            item = associate_log_line(line, line_id, split_index)

            if item.is_train_normal:
                sample.append(item)

                if len(sample) == TARGET_EVENTS:
                    break

    if len(sample) != TARGET_EVENTS:
        raise ValueError(
            f"Amostra incompleta: {len(sample)} eventos."
        )

    return sample


def catalog(parser: TemplateMiner) -> tuple:
    return tuple(
        (
            cluster.cluster_id,
            cluster.get_template(),
            cluster.size,
        )
        for cluster in sorted(
            parser.drain.clusters,
            key=lambda item: item.cluster_id,
        )
    )


def run_condition(
    sample: list[SessionLogEvent],
    prepare: PrepareFunction,
) -> tuple[dict, list[int], TemplateMiner]:
    parser = build_demo_parser()

    started = perf_counter()
    for item in sample:
        assert item.is_train_normal
        parser.add_log_message(
            prepare(item.event.content)
        )
    fit_seconds = perf_counter() - started

    before = jsonpickle.dumps(parser.drain, keys=True)
    assignments = []

    started = perf_counter()
    for item in sample:
        prepared = prepare(item.event.content)
        cluster = parser.match(
            prepared,
            full_search_strategy="always",
        )

        if cluster is None:
            raise AssertionError(
                f"Linha {item.event.line_id} não reconhecida."
            )

        assignments.append(cluster.cluster_id)

    match_seconds = perf_counter() - started
    state_preserved = (
        jsonpickle.dumps(parser.drain, keys=True) == before
    )

    clusters = list(parser.drain.clusters)
    singleton_events = sum(
        cluster.size
        for cluster in clusters
        if cluster.size == 1
    )

    report = {
        "clusters": len(clusters),
        "singleton_clusters": sum(
            cluster.size == 1 for cluster in clusters
        ),
        "events_in_singletons": singleton_events,
        "unknown_on_fit_sample": 0,
        "state_preserved": state_preserved,
        "fit_seconds": fit_seconds,
        "match_seconds": match_seconds,
    }

    return report, assignments, parser


def main() -> None:
    sample = select_sample()

    # report_a, assignments_a, parser_a = run_condition(
    #     sample,
    #     prepare_hdfs_message,
    # )
    report_b, assignments_b, parser_b = run_condition(
        sample,
        prepare_hdfs_message_candidate,
    )
    report_c, assignments_c, parser_c = run_condition(
        sample,
        prepare_hdfs_message_candidate_c,
    )

    # Repetir com parsers novos para verificar determinismo.
    # _, repeated_a, repeated_parser_a = run_condition(
    #     sample,
    #     prepare_hdfs_message,
    # )
    _, repeated_b, repeated_parser_b = run_condition(
        sample,
        prepare_hdfs_message_candidate,
    )
    _, repeated_c, repeated_parser_c = run_condition(
        sample,
        prepare_hdfs_message_candidate_c,
    )

    # assert assignments_a == repeated_a
    assert assignments_b == repeated_b
    assert assignments_c == repeated_c
    # assert catalog(parser_a) == catalog(repeated_parser_a)
    assert catalog(parser_b) == catalog(repeated_parser_b)
    assert catalog(parser_c) == catalog(repeated_parser_c)

    # changed_messages = sum(
    #     prepare_hdfs_message(item.event.content)
    #     != prepare_hdfs_message_candidate(item.event.content)
    #     for item in sample
    # )

    changed_messages = sum(
        prepare_hdfs_message_candidate(item.event.content)
        != prepare_hdfs_message_candidate_c(item.event.content)
        for item in sample
    )

    # Descobrir quais clusters de A foram reunidos em B.
    merged = defaultdict(set)
    examples = defaultdict(list)

    for item, cluster_b, cluster_c in zip(
        sample,
        assignments_b,
        assignments_c,
    ):
        merged[cluster_c].add(cluster_b)

        if len(examples[cluster_b]) < 3:
            examples[cluster_b].append({
                "line_id": item.event.line_id,
                "component": item.event.component,
                "original": item.event.content,
                "candidate": prepare_hdfs_message_candidate(
                    item.event.content
                ),
            })

    templates_c = {
        cluster.cluster_id: cluster.get_template()
        for cluster in parser_b.drain.clusters
    }

    merged_examples = [
        {
            "candidate_cluster": cluster_c,
            "baseline_clusters": sorted(clusters_b),
            "candidate_template": templates_c[cluster_c],
            "examples": examples[cluster_c],
        }
        for cluster_c, clusters_b in merged.items()
        if len(clusters_b) > 1
    ]

    result = {
        "scope": "B/C no mesmo treino; não mede generalização",
        "sample_events": len(sample),
        "changed_messages": changed_messages,
        "condition_b": report_b,
        "condition_c": report_c,
        "deterministic": True,
        "merged_cluster_examples": merged_examples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()