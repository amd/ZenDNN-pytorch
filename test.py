import re
test = '/aten/src/ATen/native/tags.yaml'
reg2 = re.compile('test/test_type_promotion.py')
reg = re.compile('/aten/src/ATen/native/tags.yaml')

print(reg2.match(test))
print(reg.match(test))