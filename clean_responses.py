import json

def clean_generations(file_path, output_path="data/cleaned_generations_20.json"):
    with open(file_path) as input_file:
        dict_gen = json.loads(input_file.read())

    for key in dict_gen:
        el = dict_gen[key]
        dict_gen[key]['cleaned_generations'] = []
        for generation in el['generations']:
            if el['prompt'] in generation:
                generation = generation[len(el['prompt']):]
                generation = generation.strip()
                dict_gen[key]['cleaned_generations'].append(generation)
        dict_gen[key]['cleaned_most_probable_answer'] = dict_gen[key]['most_probable_answer']
        if el['prompt'] in el['most_probable_answer']:
            dict_gen[key]['cleaned_most_probable_answer'] = el['most_probable_answer'][len(el['prompt']):].strip()
    
    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(dict_gen))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="File path for the generations to clean")
    
    args = parser.parse_args()
    
    clean_generations(args.file_path)