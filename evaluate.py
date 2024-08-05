from datasets import load_dataset
from transformers import EvalPrediction

from app import generate_response

dataset = load_dataset("super_glue", "record")

# 6. Evaluate model on superGLUE
def evaluate_model_on_superGLUE():
    # print("length of query: " + str(len(dataset["Query"])))
    # print("length of query: " + str(len(dataset["Query"])))
    results = []
    i = 0
    for example in dataset['validation']:
        print("index: " + str(i))
        i+=1
        message = example['passage']  # Adjust based on the specific superGLUE task structure
        generated_response = generate_response(message)
        print("length of results: " + str(len(results)))
        # print("response: " + generated_response)
        results.append({
            'input': message,
            'generated_response': generated_response,
            'reference': example['answers']  # Adjust based on the specific superGLUE task structure
        })
    
    # Here you can implement a custom evaluation metric or use existing metrics
    # This is a placeholder for evaluation logic
    print("correct number: " + str(correct))
    correct = 0
    total = len(results)
    for result in results:
        if result['generated_response'] in result['reference']:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    evaluate_model_on_superGLUE()

if __name__ == '__main__':
    main()
