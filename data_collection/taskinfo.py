import json, sys
tid = int(sys.argv[1])
t = next(x for x in json.load(open('test.raw.json', encoding='utf-8')) if x['task_id'] == tid)
print('INTENT:', t['intent'])
print('START :', t.get('start_url'))
print('EVAL  :', t['eval']['eval_types'])
print('ANSWER:', json.dumps(t['eval'].get('reference_answers', {})))