import argparse
import random
import os

from prompt.prompt_builder import PromptBuilder
from utils import *
from tqdm import tqdm

seed = os.urandom(16)
random.seed(seed)


def main(args):

    syntax_correct_count = 0
    solution_correct_count = 0

    current_example = 0
    max_retries = args.max_retries
    max_mend_retries = args.mend_retries

    learning_examples = args.learning_examples
    batch_examples = args.batch_examples
    model = args.model
    strat = args.strategy
    k = args.sample_sz
    representation = args.representation
    remove_random_percentage = args.remove_random
    remove_predicate = args.remove_predicate
    state_mending = args.state_mending
    batch_theory = args.batch_theory

    regression_examples = []

    pb = PromptBuilder('CLEVR')
    examples = pb.generate_examples(strat, k, representation, remove_predicate)
    # print(examples)
    # exit()
    # examples = json.load(open(
    #     '/home/nelson/Research/IJCAI24/logs/experiments/random_removal/select/gpt4/20240113_2107/examples_used.txt'))
    # print(examples)
    # print(type(examples), len(examples))

    # with open("correct_examples_CLEVR.json", "w") as final:
    # json.dump(examples, final)
    # exit()
    # random.shuffle(examples)

    print("Learning examples:", len(examples))
    # print(examples[0]['question'])
    # exit()
    # print(examples)
    # exit()

    incumbent_theory = ''.join(
        open('/home/nelson/Research/IJCAI24/preprompt/theory/CLEVR/perfect_theory.lp').readlines())
        # open('/home/nelson/Research/IJCAI24/preprompt/theory/CLEVR/perfect_theory.lp').readlines())
        
    if batch_theory:
        if batch_theory == 'light':
            incumbent_theory = ''.join(
                open('/home/nelson/Research/IJCAI24/preprompt/theory/CLEVR/CLEVR_light.lp').readlines())
        if batch_theory == 'medium':
            incumbent_theory = ''.join(
                open('/home/nelson/Research/IJCAI24/preprompt/theory/CLEVR/CLEVR_medium.lp').readlines())
        if batch_theory == 'heavy':
            incumbent_theory = ''.join(
                open('/home/nelson/Research/IJCAI24/preprompt/theory/CLEVR/CLEVR_heavy.lp').readlines())
                    

    if remove_random_percentage:
        incumbent_theory = remove_random_lines(
            incumbent_theory, remove_random_percentage)

    if remove_predicate:
        incumbent_theory = remove_lines_with_predicates(
            incumbent_theory, predicates=[remove_predicate])

    initial_theory = incumbent_theory

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    log_incumbent_theory(date_str,
                         incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples)

    if not learning_examples:
        learning_examples = len(examples)

    while current_example < learning_examples:
        print(current_example, learning_examples)

        if batch_examples:
            prompt = ''
            incumbent_examples = examples[current_example:min(
                current_example+batch_examples, len(examples))]
            for example in incumbent_examples:
                prompt += example['question'] + '\n#########\n'
        else:
            incumbent_examples = [examples[current_example]]
            prompt = '%Encoded Question\n' +\
                examples[current_example]['question'] + '\n' +\
                '%Encoded Scene\n' +\
                examples[current_example]['scene'] + '\n' +\
                '%Expected Answer\n' +\
                'ans({}).'.format(examples[current_example]['answer'])

        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, incumbent_examples)

        if semantic_check:
            if batch_examples:
                current_example += batch_examples
            else:
                current_example += 1
            continue

        # print(current_example)

        retries = 0
        preprompt = pb.generate_preprompt([incumbent_theory])
        # print(preprompt)
        # exit()

        while retries < max_retries:

            # print(prompt, preprompt)
            
            response = ask_LLM(prompt, preprompt, model)
            # print(response)
            response = response.replace('\\_', '_')

            # print(prompt)
            # print(response)
            # exit()

            extended_theory = incumbent_theory + '\n' + response
            syntax_check, syntax_error = check_asp_syntax(extended_theory)
            if syntax_check:
                semantic_check, semantic_error = run_asp_code(
                    extended_theory, incumbent_examples)
                if semantic_check:
                    # if args.merge_theories:
                    #     new_theory = merge_asp_encodings(
                    #         incumbent_theory, extended_theory, model)
                    #     if check_asp_syntax(new_theory):
                    #         incumbent_theory = new_theory
                    #         break

                    if args.regressive_test:
                        for example in incumbent_examples:
                            regression_examples.append(example)
                        semantic_check, semantic_errors = run_asp_code(
                            extended_theory, regression_examples)
                        if semantic_check:
                            print('Regressive Test Sucess')
                            incumbent_theory = extended_theory
                            break

                    incumbent_theory = extended_theory
                    syntax_correct_count += batch_examples
                    solution_correct_count += batch_examples
                    break

                else:
                    mend_retries = 0
                    while mend_retries < max_mend_retries:

                        # print(response, semantic_error, examples[current_example]['answer'])
                        if state_mending:
                            mended_rule = mend_semantics_with_states(
                                response, semantic_error, examples[current_example]['answer'], incumbent_theory, preprompt, model)
                        else:
                            mended_rule = mend_semantics(
                                response, semantic_error, examples[current_example]['answer'], incumbent_theory, preprompt, model)
                        # print(mended_rule)
                        # exit()
                        mended_rule = mended_rule.replace('\\_', '_')

                        extended_theory = incumbent_theory + '\n' + mended_rule
                        syntax_check, error = check_asp_syntax(extended_theory)
                        if syntax_check:
                            semantic_check, error = run_asp_code(
                                extended_theory, incumbent_examples)
                            if semantic_check:

                                if args.regressive_test:
                                    for example in incumbent_examples:
                                        regression_examples.append(example)
                                    semantic_check, semantic_errors = run_asp_code(
                                        extended_theory, regression_examples)
                                    if semantic_check:
                                        print('Regressive Test Sucess')
                                        incumbent_theory = extended_theory
                                        break

                                incumbent_theory = extended_theory
                                syntax_correct_count += batch_examples
                                solution_correct_count += batch_examples
                                break
                        mend_retries += 1

            else:
                mend_retries = 0
                while mend_retries < max_mend_retries:
                    # print('initial rules', response)
                    # print('ERROR')
                    # print('error', syntax_error)
                    mended_rule = mend_syntax(
                        response, syntax_error, preprompt, model)
                    mended_rule.replace('\\_', '_')
                    # print('mended_rule', mended_rule)
                    # exit()
                    extended_theory = incumbent_theory + '\n' + mended_rule
                    syntax_check, error = check_asp_syntax(extended_theory)
                    if syntax_check:
                        semantic_check, error = run_asp_code(
                            extended_theory, incumbent_examples)
                        if semantic_check:

                            if args.regressive_test:
                                for example in incumbent_examples:
                                    regression_examples.append(example)
                                semantic_check, semantic_errors = run_asp_code(
                                    extended_theory, regression_examples)
                                if semantic_check:
                                    print('Regressive Test Sucess')
                                    incumbent_theory = extended_theory
                                    break

                            incumbent_theory = extended_theory
                            syntax_correct_count += batch_examples
                            solution_correct_count += batch_examples
                            break
                    mend_retries += 1

            retries += 1

        if batch_examples:
            current_example += batch_examples
        else:
            current_example += 1

        log_incumbent_theory(date_str,
                             incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples)

    # if incumbent_theory == initial_theory:
    #     print('Last Theory is the same as Initial Theory')
    #     write_to_log('./logs/'+date_str+'/config_log.txt',
    #                  seed, args, 'SAME AS INPUT')
    #     exit()
    # else:
    #     print('Obtained a different Theory')

    log_incumbent_theory(date_str,
                         incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples)

    final_correct_train = 0
    current_example = 0

    print('Testing Theory on Learning Examples')
    while current_example < learning_examples:
        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, [examples[current_example]])
        if semantic_check:
            final_correct_train += 1
        current_example += 1

    print(f"Final Training Correct Solutions: {final_correct_train}")
    print(f"Total Training Examples: {learning_examples}")

    test_suite = open('./dataset/CLEVR/test_suite.json')
    test_suite = json.load(test_suite)
    learning_examples = len(test_suite)
    final_correct_test = 0

    print('Validating Theory')

    for i in tqdm(range(learning_examples)):
        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, [test_suite[i]])
        if semantic_check:
            final_correct_test += 1

    print(f"Final Correct Solutions: {final_correct_test}")
    print(f"Total Solutions: {learning_examples}")
    print(f"Percentage: {final_correct_test/learning_examples*100}")

    # print('Validating Initial Theory')

    # final_correct_initial_theory = 0

    # if args.validate:
    # for i in tqdm(range(learning_examples)):
    #     x,y = run_asp_code(initial_theory, [test_suite[i]])
    #     if x:
    #         final_correct_initial_theory += 1
    # else:
    #     final_correct_initial_theory = -1

    # print(f"Final Initial Theory Solutions: {final_correct_initial_theory}, {final_correct_initial_theory/learning_examples*100}")
    
    write_to_log('./logs/'+date_str+'/config_log.txt', seed, args, f"Final Training Correct Solutions: {
                 final_correct_train}", f"Final Correct Solutions: {final_correct_test}", f"Average: {round(final_correct_test/learning_examples*100,2)}")
    save_examples('./logs/'+date_str+'/examples_used.txt', examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASP Encoding and LLM Interaction")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Maximum number of retries")
    parser.add_argument("--mend_retries", type=int, default=1,
                        help="Maximum number of mending retries")
    parser.add_argument("--learning_examples", type=int,
                        default=0, help="Number of learning examples")
    parser.add_argument("--batch_examples", type=int,
                        default=0, help="Number of batch examples")
    parser.add_argument("--model", type=str,
                        default="gpt-4-1106-preview", help="LLM model to be used")
    parser.add_argument("--strategy", type=str, default="pred",
                        help="Strategy used to sample examples")
    parser.add_argument("--sample_sz", type=int, default=10,
                        help="Sample size for the strategy selected")
    parser.add_argument("--regressive_test", default=True, type=bool,
                        help="Use regressive testing of previous examples")
    parser.add_argument("--representation", type=str, default="flat",
                        help="Representation to be used")
    parser.add_argument("--remove_random", type=int, default=0,
                        help="Remove a percentage of random lines from the perfect theory to use as initial theory.")
    parser.add_argument("--remove_predicate", type=str, default=None,
                        help="Remove any rule where the predicated selected appears in the perfect theory and use this as initial theory.")
    parser.add_argument("--state_mending", type=bool, default=False,
                        help="Use semantic mending with states of the program.")
    parser.add_argument("--batch_theory", type=str, default="",
                        help="Which batch theory to use.")
    # parser.add_argument("--no_examples", type=bool, default=False,
    #                     help="Pass only the theory and autocomplete inmediately.")
    # parser.add_argument("--question_type", type=str,
    #                     help="Select a subset of questions from the dataset")

    args = parser.parse_args()
    main(args)
