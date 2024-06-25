text = 0

def set_global(var):
    global text
    text = var

print(text)  # 0 출력
set_global(3)
print(text)  # 3 출력