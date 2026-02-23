

class User:

    def __init__(self, username= None, passsword= None, email= None, birthday= None):
        self.username = username
        self.password = passsword
        self.email = email
        self.birthday = birthday



if __name__ == "__main__":
    pete = User(username= "Peter", passsword= "Password",
    email= "pete@some.com", birthday= "2025-01-01")
    print(pete)
    #print(help(User))

