import os
import base64
import json
import random
from PIL import Image
import io
from litellm import completion
import argparse

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "sk-None-jFbxGP93PDk2Yk1vo63ST3BlbkFJQtK4HzVLPQ17LsRxkjaI"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_question(image_base64, question, possible_answers):
    """Generates a response from GPT-4 model about the image."""
    messages = [
        {"role": "system", "content": "You are an AI that can specializes in Visual Question Answering. You must choose an answer from the possible answers list. Only output the chosen answer and nothing more. The answer must be in lowercase."},
        {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"}
                    }
        ]},
        {"role": "user", "content": question},
        {"role": "user", "content": f'List of possible answers: {possible_answers}'}
    ]
    
    response = completion(
        model="openai/gpt-4o",
        messages=messages
    )
    return response

def main(dataset):
    if dataset == 'GQA':
        questions_file_path = '/home/nhiguera/Research/KR2024/GQA/questions/val_balanced_questions.json'

        with open(questions_file_path, 'r') as file:
            question_data = json.load(file)

        random_question_keys_sample = random.sample(list(question_data.keys()), 1000)

        gqa_answers_file = open("gqa_answers.txt", "r")
        possible_answers = gqa_answers_file.read().replace('\n', ' ').split(".")

        possible_answers = []
        for key in random_question_keys_sample:
            possible_answers.append(question_data[key]['answer'])

        possible_answers = list(set(possible_answers))
        possible_answers.sort()
        possible_answers = ', '.join(possible_answers)
        print(possible_answers)

        correct = 0
        total = 0
        for key in random_question_keys_sample:
            try:
                # Path to the image file
                image_path = f"/home/nhiguera/Research/KR2024/GQA/images/{question_data[key]['imageId']}.jpg"

                # Question about the image
                question = question_data[key]['question']

                # Encode the image to base64
                image_base64 = encode_image(image_path)

                # Generate the responses
                response = generate_question(image_base64, question, possible_answers)

                # Print the response
                print('imageid', question_data[key]['imageId'])
                print('question', question_data[key]['question'])
                print('gt answer', question_data[key]['answer'])
                print('response', response.choices[0].message.content)
                total += 1
                if response.choices[0].message.content.lower() == question_data[key]['answer']:
                    correct += 1
            except:
                continue

        accuracy = correct / total * 100
        print(correct, total, accuracy)

        with open(f'{dataset}_stats.txt', 'w') as stats_file:
            stats_file.write(f"Correct: {correct}\nTotal: {total}\nAccuracy: {accuracy}%\n")

    else:
        questions_file_path = '/home/nhiguera/Research/KR2024/CLEVR/questions/CLEVR_val_questions.json'

        with open(questions_file_path, 'r') as file:
            question_data = json.load(file)

        random_question_sample = random.sample(list(question_data["questions"]), 1000)

        clevr_answers_file = open("clevr_answers.txt", "r")
        possible_answers = clevr_answers_file.read().replace('\n', ' ').split(".")

        possible_answers = []
        for question in random_question_sample:
            possible_answers.append(question['answer'])

        possible_answers = list(set(possible_answers))
        possible_answers.sort()
        possible_answers = ', '.join(possible_answers)
        print(possible_answers)

        correct = 0
        total = 0
        for question in random_question_sample:
            # Path to the image file
            try:
                image_path = f"/home/nhiguera/Research/KR2024/CLEVR/images/val/{question['image_filename']}"

                # Encode the image to base64
                image_base64 = encode_image(image_path)

                # Generate the responses
                response = generate_question(image_base64, question['question'], possible_answers)

                # Print the response
                print('imageid', question['image_filename'])
                print('question', question['question'])
                print('gt answer', question['answer'])
                print('response', response.choices[0].message.content)
                total += 1
                if response.choices[0].message.content.lower() == question['answer']:
                    correct += 1
            except:
                continue

        accuracy = correct / total * 100
        print(correct, total, accuracy)

        with open(f'{dataset}_stats.txt', 'w') as stats_file:
            stats_file.write(f"Correct: {correct}\nTotal: {total}\nAccuracy: {accuracy}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='Dataset to use: GQA or CLEVR', default='CLEVR')
    args = parser.parse_args()
    main(args.dataset)