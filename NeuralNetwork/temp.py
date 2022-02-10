import json
with open('models/models.json', 'r') as infile:
    json_object = json.load(infile)

found = False
for model in json_object['models']:
    if(model['name'] == "default"):
        model['acc'] = 100
        found = True
        break
    
if not found:
    json_object['models'].append({'name': 'default', 'acc': 100})

print(json_object)
