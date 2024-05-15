import argparse
import random
import os
import sys

from tqdm import tqdm

sys.path.append('/home/nelson/Research/IJCAI24/prompt')

from parse_GQA_ILASP import *
from prompt_builder import PromptBuilder
from utils import *

pb = PromptBuilder('GQA')
remove_predicate = 'exist'
examples = pb.generate_examples('pred', 100, 'flat', remove_predicate)
incumbent_theory = ''.join(
open('/home/nelson/Research/IJCAI24/preprompt/theory/GQA/perfect_theory.lp').readlines())

if remove_predicate:
    incumbent_theory = remove_lines_with_predicates(
        incumbent_theory, predicates=[remove_predicate])

initial_theory = incumbent_theory

ilasp_examples = []
for example in examples:
    ilasp_ex = create_ilasp_input(example)
    ilasp_examples.append(ilasp_ex)

with open('ilasp/exs.pl', 'w') as file: file.write('\n\n'.join(ilasp_examples))


# ilasp_bk = '''
# % Config
# #bias("allow_recursive.").

# #minhl(1).  % Minimum rule head length
# #maxhl(1).  % Maximum rule head length
# #maxv(5).   % Maximum number of variables per rule

# % Rule heads

# #modeh(bool(var(state_id), const(ans))).

# #modeb(exist(var(state_id), var(state_id))).
# #modeb(state(var(state_id), var(obj_id))).
# #modeb(bool(var(state_id), const(ans))).

# #constant(ans, yes).
# #constant(ans, no).

# % Theory

# '''

# ilasp_bk += initial_theory

# with open('ilasp/bk.pl', 'w') as file: file.write(ilasp_bk)