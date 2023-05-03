from multiprocessing import Process

import ComplexKalman

if __name__ == "__main__":
    p1 = Process(target=ComplexKalman.run)
    p2 = Process(target=ComplexKalman.run)

    print("Starting p1")
    p1.start()
    print("Starting p2")
    p2.start()

    print("Started both")
