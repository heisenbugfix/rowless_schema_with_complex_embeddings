s = set()
with open('../data/kb_test_35_cap10','r') as f:
	for each in f:
		s.update(each.split('\t')[-1].split(' '))
for each in s:
	print(each)