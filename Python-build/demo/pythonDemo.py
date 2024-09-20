import demoClass #import the module
import subprocess

def tests():
    #constructor test
    DUT = demoClass.DemoClass(1,2)
    DUT2 = demoClass.DemoClass(1,-2)
    DUT3 = demoClass.DemoClass(123,51)

    #addition test
    print("\nTesting addition...")
    print(f"DUT.add(): Expected/actual=  3|{DUT.add()}")
    print(f"DUT2.add(): Expected/actual= -1|{DUT2.add()}")
    print(f"DUT3.add(): Expected/actual= 174|{DUT3.add()}")

    #subtraction test
    print("\nTesting subtraction...")
    print(f"DUT.subtract(): Expected/actual= -1|{DUT.subtract()}")
    print(f"DUT2.subtract(): Expected/actual= 3|{DUT2.subtract()}")
    print(f"DUT3.subtract(): Expected/actual= 72|{DUT3.subtract()}")

    #multiplication test
    print("\nTesting multiplication...")
    print(f"DUT.multiply(): Expected/actual= 2|{DUT.multiply()}")
    print(f"DUT2.multiply(): Expected/actual= -2|{DUT2.multiply()}")
    print(f"DUT3.multiply(): Expected/actual= 6273|{DUT3.multiply()}")

    #division test
    print("\nTesting division...")
    print(f"DUT.divide(): Expected/actual= 0.5|{DUT.divide()}")
    print(f"DUT2.divide(): Expected/actual= -0.5|{DUT2.divide()}")
    print(f"DUT3.divide(): Expected/actual= 2.411764705882353|{DUT3.divide()}")

    #getters test
    print("\nTesting getters...")
    print(f"DUT.getA(): Expected/actual= 1|{DUT.getA()}")
    print(f"DUT.getB(): Expected/actual= 2|{DUT.getB()}")
    print(f"DUT2.getA(): Expected/actual= 1|{DUT2.getA()}")
    print(f"DUT2.getB(): Expected/actual= -2|{DUT2.getB()}")
    print(f"DUT3.getA(): Expected/actual= 123|{DUT3.getA()}")
    print(f"DUT3.getB(): Expected/actual= 51|{DUT3.getB()}")

    #setters test
    print("\nTesting setters...")
    DUT.setA(3)
    DUT.setB(4)
    print(f"DUT.getA(): Expected/actual= 3|{DUT.getA()}")
    print(f"DUT.getB(): Expected/actual= 4|{DUT.getB()}")
    try:
        DUT.setA("a")
        print("Fail")
    except:
        print("Pass")
    try:
        DUT.setB(2.4)
        print("Fail")
    except:
        print("Pass")

    #iseven test
    print("\nTesting isEven...")
    print(f"DUT.isEven(): Expected/actual= False|{DUT.isEven()}")
    print(f"DUT2.isEven(): Expected/actual= True|{DUT2.isEven()}")

    #fibonacci test
    print("\nTesting fibonacci...")
    DUT.setA(1)
    DUT.setB(200)
    print(f"DUT.fib(): from {DUT.getA()} to {DUT.getB()}:")
    DUT.calcFibAB()

    #test maps
    # print("\nTesting maps...")
    # result = DUT.getMap()
    # print(f"DUT.mapAB(): {type(result)}|{result}")

    # donut test (fps, distane, refresh rate, xpos,ypos)
    # DUT.dounut(60,5, .2,0, 40, 12)
    DUT.dounut(30,7, .4, 100, 40, 12)



    

if __name__ == "__main__":
    #execute python setup.py build_ext --inplace
    result = subprocess.run(
    ['python3', 'setup.py', 'build_ext', '--inplace'],
    capture_output=True,  # Capture standard output and error
    text=True              # Decode output as a string
)
    tests()
    print("All tests Ran")