from data_generator import generate_data_from_product

set1 = generate_data_from_product('1121355', 'Lucha Libre AAA: Heroes Del Ring - Xbox 360')

for item in set1:
	print item, '\n'