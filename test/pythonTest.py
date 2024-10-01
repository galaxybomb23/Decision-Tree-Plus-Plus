import sys
import re
# Test the DecisionTree utility class

def cleanup_csv(line):
    # remove special charactersx    
    line = line.translate(bytes.maketrans(b":?/()[]{}'",b"          ")) \
           if sys.version_info[0] == 3 else line.translate(string.maketrans(":?/()[]{}'","          "))
    
    # remove double quoted text
    double_quoted = re.findall(r'"[^\"]+"', line[line.find(',') : ])
    for item in double_quoted:
        clean = re.sub(r',', r'', item[1:-1].strip())
        parts = re.split(r'\s+', clean.strip())
        line = str.replace(line, item, '_'.join(parts))
    
    # remove whitespace
    white_spaced = re.findall(r',(\s*[^,]+)(?=,|$)', line)
    for item in white_spaced:
        litem = item
        litem = re.sub(r'\s+', '_', litem)
        litem = re.sub(r'^\s*_|_\s*$', '', litem) 
        line = str.replace(line, "," + item, "," + litem) if line.endswith(item) else str.replace(line, "," + item + ",", "," + litem + ",") 
    fields = re.split(r',', line)
    
    # generate new string
    newfields = []
    for field in fields:
        newfield = field.strip()
        if newfield == '':
            newfields.append('NA')
        else:
            newfields.append(newfield)
    line = ','.join(newfields)
    return line

#tests
tests = {
    ("hello, world, this, is, a, test", "hello,world,this,is,a,test"),
        ("hello, , world, , test", "hello,NA,world,NA,test"),
        ('hello, "quoted, text", world', "hello,quoted_text,world"),
        ("hello, world, this, is, a, test, ", "hello,world,this,is,a,test,NA"),
        ("hello, world, this, is, a, test, ,", "hello,world,this,is,a,test,NA,NA"),
}

for test in tests:
    assert cleanup_csv(test[0]) == test[1]
    print("Test passed")