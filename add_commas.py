with open('results/train.json', 'r') as istr:
    with open('results/train_new.json', 'w') as ostr:
        for i, line in enumerate(istr):
            # Get rid of the trailing newline (if any).
            line = line.rstrip('\n')
            line += ','
            print(line, file=ostr)
