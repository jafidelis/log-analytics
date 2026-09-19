from importlib.metadata import version

from log_analytics.data.prepare_hdfs_v1 import prepare_hdfs_message
from log_analytics.data.fit_hdfs_v1 import build_demo_parser


def main() -> None:
    print(f"Drain3: {version('drain3')}")

    parser = build_demo_parser()

    messages = [
        (
            "Receiving block blk_1 "
            "src: /10.0.0.1:54106 dest: /10.0.0.2:50010"
        ),
        (
            "Receiving block blk_2 "
            "src: /10.0.0.3:54106 dest: /10.0.0.4:50010"
        ),
        (
            "Receiving block blk_3 "
            "src: /10.0.0.5:40524 dest: /10.0.0.6:50010"
        ),
        "Completed block blk_4",
    ]

    for number, original in enumerate(messages, start=1):
        prepared = prepare_hdfs_message(original)
        result = parser.add_log_message(prepared)

        print(f"\nMensagem {number}")
        print(f"Original:  {original}")
        print(f"Preparada: {prepared}")
        print(f"Alteração: {result['change_type']}")
        print(f"Cluster:   {result['cluster_id']}")
        print(f"Tamanho:   {result['cluster_size']}")
        print(f"Template:  {result['template_mined']}")

    print("\nCatálogo ao final:")
    for cluster in sorted(
        parser.drain.clusters,
        key=lambda item: item.cluster_id,
    ):
        print(
            f"ID={cluster.cluster_id} "
            f"mensagens={cluster.size} "
            f"template={cluster.get_template()}"
        )


if __name__ == "__main__":
    main()