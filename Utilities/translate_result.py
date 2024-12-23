def translate_wasd(lst):
    output = ''
    if lst[0] == 1:
        output += 'w'
    if lst[1] == 1:
        output += 'a'
    if lst[2] == 1:
        output += 's'
    if lst[3] == 1:
        output += 'd'
    if len(output) == 0:
        output = ''
    return output
