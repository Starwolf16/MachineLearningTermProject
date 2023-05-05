# Convert RGB color components to a single number

#take RGB Values for input
r = int(input("Red? "))
g = int(input("Green? "))
b = int(input("Blue? "))

#perform bit shifts and bitwise OR
color_num = (r << 16) | (g << 8) | b

#print resulting number
print(color_num)
