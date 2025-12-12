with open("sample.txt", "w") as f:
    f.write("Hello Charan!\n")
    f.write("Welcome to Python Full Stack.\n")

with open("sample.txt", "r") as f:
    content = f.read()

print("File Content:")
print(content)
