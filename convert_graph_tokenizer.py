import os
import sys
import random
from tqdm import tqdm
from special_tokens import new_tokens_amr
from init_tokenizer import CustomT5Tokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))

folder = sys.argv[1]


from transformers import T5Tokenizer

tokenizer = CustomT5Tokenizer.from_pretrained("t5-base")
new_tokens_vocab = {}
new_tokens_vocab["additional_special_tokens"] = []
for idx, t in enumerate(new_tokens_amr):
    new_tokens_vocab["additional_special_tokens"].append(t)
num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
tokenizer.unique_no_split_tokens.sort(key=lambda x: -len(x))
# print(self.tokenizer.unique_no_split_tokens)
print("We have added tokens", num_added_toks)


def open_file(f):
    return open(f, "r").readlines()


def convert_graph(file_source, file_graph):
    mappings = []
    amr_toks = []
    amr_toks_len = []

    for l_idx, line in enumerate(file_source):
        line = line.strip()
        tokens = line.split()  # Adjust tokenizer as necessary
        tokenized_tokens = [tokenizer.tokenize(token) for token in tokens]
        flat_token_list = [item for sublist in tokenized_tokens for item in sublist]
        amr_toks_len.append(len(flat_token_list))
        amr_toks.append(" ".join(flat_token_list))

        # Create mappings
        map = {}
        idx_original = 0
        current_idx = 0
        for token_group in tokenized_tokens:
            map[idx_original] = list(range(current_idx, current_idx + len(token_group)))
            current_idx += len(token_group)
            idx_original += 1

        mappings.append(map)

    # Create new graphs based on the mappings
    new_graphs = []
    for l_idx, line in enumerate(file_graph):
        src_graphs = line.strip().split()
        new_edges = []
        for edge in src_graphs:
            parts = edge.strip("()").split(",")
            e1, e2, rel = int(parts[0]), int(parts[1]), parts[2]

            # Map edges based on specific connection logic
            mapped_e1 = mappings[l_idx][e1]
            mapped_e2 = mappings[l_idx][e2]

            # Internal sequential links for each token
            for i in range(len(mapped_e1) - 1):
                new_edges.append(f"({mapped_e1[i]},{mapped_e1[i+1]},seq)")
            for i in range(len(mapped_e2) - 1):
                new_edges.append(f"({mapped_e2[i]},{mapped_e2[i+1]},seq)")

            # Adjust cross-token relationship to connect the last of previous to the first of next
            if mapped_e1 and mapped_e2:
                new_edges.append(f"({mapped_e1[-1]},{mapped_e2[0]},{rel})")

        new_graphs.append(" ".join(new_edges))

    return new_graphs, amr_toks_len


graph_files = ["train.graph", "test.graph", "val.graph"]
source_files = ["train.source", "test.source", "val.source"]

for s, g in zip(source_files, graph_files):
    print(g)
    s_data = open_file(folder + s)
    g_data = open_file(folder + g)
    new_graph, amr_toks_len = convert_graph(s_data, g_data)

    new_graph_file = open(folder + g + ".tok", "w")
    for ng in new_graph:
        new_graph_file.write(ng + "\n")
