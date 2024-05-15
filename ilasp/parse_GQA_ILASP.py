def parse_scene(scene_text):
    # This function parses the scene information into a clean, formatted string
    # that can be used in ILASP syntax. Atoms are separated by dots.
    scene_lines = scene_text.strip().split('\n')
    cleaned_scene = ' '.join(line.strip() for line in scene_lines if line)
    return cleaned_scene

def parse_question(question_text):
    # This function parses the question information into a clean, formatted string
    # that can be used in ILASP syntax. Atoms are separated by dots.
    question_lines = question_text.strip().split('\n')
    cleaned_question = ' '.join(line.strip() for line in question_lines if line)
    return cleaned_question

def create_ilasp_input(data):
    # This function creates the ILASP formatted input based on the given dictionary.
    question = parse_question(data['question'])
    scene = parse_scene(data['scene'])
    answer = data['answer']

    ilasp_example = f"#pos({{ans({answer})}}, {{}}, {{{question} {scene}}})."
    return ilasp_example

# Example data dictionary
# data = {
#     'question': 'scene(0).\nselect(1, 0, wheel).\nunique(2, 1).\nverify_attr(3, 2, color, green).\nend(3).',
#     'scene': 'object(872226).\nhas_attr(872226, class, cone).\nhas_attr(872226, name, cone).\nhas_attr(872226, class, object).\nhas_attr(872226, color, orange).\nhas_attr(872226, any, safety).\nhas_attr(872226, hposition, right).\nhas_attr(872226, vposition, bottom).\nhas_attr(872226, vposition, middle).\nobject(872249).\nhas_attr(872249, class, airport).\nhas_attr(872249, name, airport).\nhas_attr(872249, class, place).\nhas_attr(872249, hposition, middle).\nhas_attr(872249, vposition, middle).\nobject(872248).\nhas_attr(872248, class, wheel).\nhas_attr(872248, name, wheel).\nhas_attr(872248, color, black).\nhas_attr(872248, hposition, left).\nhas_attr(872248, vposition, bottom).\nhas_attr(872248, vposition, middle).\nobject(872245).\nhas_attr(872245, class, sky).\nhas_attr(872245, name, sky).\nhas_attr(872245, class, nature_environment).\nhas_attr(872245, color, blue).\nhas_attr(872245, hposition, middle).\nhas_attr(872245, vposition, top).',
#     'answer': 'no'
# }

# Generating the ILASP input
# ilasp_input = create_ilasp_input(data)
# print(ilasp_input)