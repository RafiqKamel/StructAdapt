from transformers import T5Tokenizer
from special_tokens import new_tokens_amr
from seq2seq_dataset import Seq2SeqDataset
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


def create_dataloader(
    data_path: str,
    batch_size: int = 128,
    model_name: str = "t5-base",
    max_input_length: int = 384,
    max_output_length: int = 384,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    new_tokens_vocab = {}
    new_tokens_vocab["additional_special_tokens"] = []
    for idx, t in enumerate(new_tokens_amr):
        new_tokens_vocab["additional_special_tokens"].append(t)
    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
    dataset = Seq2SeqDataset(
        tokenizer,
        data_path,
        max_source_length=max_input_length,
        max_target_length=max_output_length,
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=0,
    )
    return dl, tokenizer


def generate_amr_string(node_labels, G):
    edges = []
    result_string = ""
    new_id_mapping = {}
    id_counter = 0
    for n in G.nodes:
        succ = list(G.successors(n))
        random.shuffle(succ)
        curr_word = node_labels[n] if n in node_labels else "<END>"
        if n in new_id_mapping:
            curr_id = new_id_mapping[n]
        else:
            new_id_mapping[n] = id_counter
            curr_id = id_counter
            id_counter += 1
        for s in succ:
            if s in new_id_mapping:
                succ_id = new_id_mapping[s]
            else:
                new_id_mapping[s] = id_counter
                succ_id = id_counter
                id_counter += 1
            word = node_labels[s] if s in node_labels else "<END>"
            edges.append((curr_word, curr_id, word, succ_id))
    for e in edges:
        result_string += f"{e[0]} /{e[1]} > {e[2]} /{e[3]}\n"
    return result_string


def convert_AMR(input_ids, edges, edge_types, target_labels, tokenizer, target_text):
    # Create a directed graph
    G = nx.DiGraph()
    # Add edges and edge attributes
    labels_dict = {}
    for j in range(edges.shape[1]):
        if edge_types[j] == 1:
            continue
        src, dest = edges[:, j]
        G.add_edge(int(dest), int(src))
    labels_dict = {
        node_id: tokenizer.decode(input_ids[node_id]) for node_id in G.nodes()
    }
    AMR = {}
    AMR["graph"] = G
    AMR["node_labels"] = labels_dict
    AMR["target_labels"] = target_labels
    AMR["target_text"] = target_text
    AMR["input_ids"] = np.array(
        [iid for i, iid in enumerate(input_ids) if i in G.nodes()]
    )
    AMR["graph_string"] = generate_amr_string(labels_dict, G)

    return AMR


def draw_AMR(AMR):
    # Generate positions for a tree layout
    G = AMR["graph"]
    labels_dict = AMR["node_labels"]
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(120 * 2, 80 * 2))  # You can adjust the size as needed
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1000)
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrowstyle="-|>", arrowsize=150)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=150)
    plt.show()


# path = "./StructAdapt/data/processed_amr"
# dl, tokenizer = create_dataloader(path)
# first_batch = next(iter(dl))
# i = 1
# tgt_labels = first_batch["labels"][i]
# first_batch["input_ids"][i]
# edges, edge_type = first_batch["graphs"][i]
# AMR = convert_AMR(first_batch["input_ids"][i], edges, edge_type, tgt_labels, tokenizer)
# draw_AMR(AMR)
