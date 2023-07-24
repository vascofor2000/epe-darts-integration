""" Network architecture visualizer using graphviz """
import sys

from epe_darts import genotypes as gt


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} GENOTYPE".format(sys.argv[0]))

    genotype_str = sys.argv[1]
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    gt.plot(genotype.normal, "normal")
    gt.plot(genotype.reduce, "reduction")
