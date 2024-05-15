import json 
from pipeline.utils import sanitize, sanitize_asp

with open('../data/metadata/gqa_all_attribute.json') as f:
    all_attributes = json.load(f)

with open('../data/metadata/gqa_all_class.json') as f:
    all_categories = json.load(f)

def extract_attributes(question):
    attributes = set()
    standalone_values = set()
    for operation in question['semantic']:
        op = operation['operation']
        if op == 'relate':
            relation_type = operation['argument'].split(',')[1]
            if relation_type.startswith('same '):
                attributes.add(relation_type[5:])
        elif op == 'query':
            attributes.add(operation['argument'])
        elif op == 'common':
            attributes.update(all_attributes.keys())
        elif (op == 'same' or op == 'different') and \
             operation['argument'] != 'type':
            attributes.add(operation['argument'])
        elif op == 'filter':
            if operation['argument'].startswith('not('):
                standalone_values.add(operation['argument'][4:-1])
            else:
                standalone_values.add(operation['argument'])
        elif op == 'choose':
            standalone_values.add(operation['argument'].split('|')[0])
            standalone_values.add(operation['argument'].split('|')[1])
        elif operation['operation'] == 'verify':
            standalone_values.add(operation['argument'])
        elif op.startswith('filter') or \
             (op.startswith('verify') and op != 'verify rel') or \
             op.startswith('same') or \
             op.startswith('different'):
            attributes.add(' '.join(op.split(' ')[1:]))
        elif op.startswith('choose') and op != 'choose rel':
            if operation['argument'] == '':
                op_tokens = op.split(' ')
                if len(op_tokens) >= 3:
                    standalone_values.add(op_tokens[2])
                else:
                    token = sanitize_asp(op_tokens[1])
                    if token.endswith('er'):
                        token = token[:-2]
                        if token.endswith('i'):
                            token = token[:-1] + 'y'
                    standalone_values.add(token)
            else:
                attributes.add(' '.join(op.split(' ')[1:]))
        
    return {sanitize(a) for a in attributes if a != 'name' and a != 'vposition' and a != 'hposition'}, \
           {sanitize(v) for v in standalone_values}


def extract_classes(question):
    classes = {
        "categories": set(),
        "classes": set(),
        "all": False
    }

    def add_class_or_category(c):
        c = sanitize_asp(c)
        if c in all_categories.keys():
            classes["categories"].add(c)
        else: 
            classes["classes"].add(c)
            
    for operation in question['semantic']:
        if operation['operation'] == 'select':
            add_class_or_category(operation['argument'].split('(')[0])
        elif operation['operation'] == 'relate':
            target_class = operation['argument'].split(',')[0]
            if target_class != '_':
                add_class_or_category(target_class)
            else: 
                classes["all"] = True
        elif operation['operation'] == 'choose rel':
            add_class_or_category(operation['argument'].split(',')[0])
        elif operation['operation'] == 'verify rel':
            add_class_or_category(operation['argument'].split(',')[0])
    return classes


def extract_relations(question):
    relations = set()
    for operation in question['semantic']:
        if operation['operation'] == 'relate':
            relation = operation['argument'].split(',')[1]
            if not relation.startswith('same '):
                relations.add(relation)
        elif operation['operation'] == 'choose rel':
            relations.add(operation['argument'].split(',')[1].split('|')[0])
            relations.add(operation['argument'].split(',')[1].split('|')[1])
        elif operation['operation'] == 'verify rel':
            relations.add(operation['argument'].split(',')[1])
    return {sanitize(r) for r in relations}