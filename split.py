import json, random

def split_dataset(filename, array):
    # Open the file in write mode
    with open(filename, 'w') as file:
        # Use json.dump to write the array to the file
        json.dump(array, file)   


with open('./correct_examples_CLEVR.json') as f:
    questions = json.load(f)

l = len(questions)
ltrain = int(l*0.75)
random.shuffle(questions)

train_suite = questions[:ltrain]
test_suite = questions[ltrain:]

print(len(train_suite))
print(len(test_suite))

with open("train_suite.json", "w") as final:
    json.dump(train_suite, final)

with open("test_suite.json", "w") as final:
    json.dump(test_suite, final)