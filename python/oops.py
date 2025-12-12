class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print("Name:", self.name)
        print("Age:", self.age)

p = Person("Charan", 23)
p.info()
