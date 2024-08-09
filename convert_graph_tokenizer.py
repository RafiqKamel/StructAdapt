import os
import sys
import random
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))

folder = sys.argv[1]

new_tokens_amr = set(
    [
        ":example",
        ":range",
        ":d",
        ":degree-of",
        ":polarity",
        ":path",
        ":prep-from",
        ":source",
        ":purpose-of",
        ":frequency",
        ":ARG4-of",
        ":medium-of",
        ":consist-of",
        ":instrument",
        ":value-of",
        ":prep-to",
        ":)",
        ":op24",
        ":-P",
        ":quant-of",
        ":prep-on",
        ":location-of",
        ":purpose",
        ":op3",
        ":s",
        ":op20",
        ":time-of",
        ":snt1",
        ":ARG1-of",
        ":destination-of",
        ":ord",
        ":concession-of",
        ":century",
        ":medium",
        ":prep-against",
        ":op18",
        ":op21",
        ":prep-along-with",
        ":manner-of",
        ":ARG9",
        ":op13",
        ":frequency-of",
        ":topic-of",
        ":prep-among",
        ":snt10",
        ":degree",
        ":condition",
        ":season",
        ":op10",
        ":op7",
        ":op8",
        ":polarity-of",
        ":ARG8",
        ":prep-as",
        ":ord-of",
        ":prep-in",
        ":snt8",
        ":name-of",
        ":topic",
        ":prep-with",
        ":calendar",
        ":path-of",
        ":wiki",
        ":accompanier-of",
        ":ARG3-of",
        ":ARG3",
        ":op17",
        ":snt3",
        ":ARG5-of",
        ":era",
        ":part-of",
        ":op1-of",
        ":weekday",
        ":subset",
        ":mode",
        ":ARG2-of",
        ":age-of",
        ":op9",
        ":quarter",
        ":op1",
        ":ARG2",
        ":direction",
        ":name",
        ":polite",
        ":snt9",
        ":op11",
        ":op22",
        ":condition-of",
        ":duration-of",
        ":prep-toward",
        ":snt11",
        ":prep-for",
        ":prep-into",
        ":op2",
        ":op19",
        ":li",
        ":ARG1",
        ":op14",
        ":month",
        ":prep-under",
        ":ARG6-of",
        ":prep-on-behalf-of",
        ":prep-by",
        ":ARG6",
        ":value",
        ":day",
        ":extent",
        ":source-of",
        ":concession",
        ":example-of",
        ":subevent",
        ":destination",
        ":poss-of",
        ":snt2",
        ":op5",
        ":op16",
        ":ARG7",
        ":prep-in-addition-to",
        ":op12",
        ":ARG5",
        ":part",
        ":duration",
        ":subevent-of",
        ":manner",
        ":mod",
        ":beneficiary-of",
        ":beneficiary",
        ":op6",
        ":dayperiod",
        ":prep-at",
        ":year2",
        ":accompanier",
        ":location",
        ":snt6",
        ":age",
        ":domain",
        ":ARG0",
        ":conj-as-if",
        ":ARG4",
        ":P",
        ":prep-amid",
        ":unit",
        ":ARG7-of",
        ":snt4",
        ":op23",
        ":poss",
        ":extent-of",
        ":decade",
        ":year",
        ":direction-of",
        ":quant",
        ":op4",
        ":prep-without",
        ":scale",
        ":subset-of",
        ":snt7",
        ":prep-out-of",
        ":time",
        ":instrument-of",
        ":timezone",
        ":snt5",
        ":op15",
        ":ARG0-of",
    ]
)

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
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
