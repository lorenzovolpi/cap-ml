import traceback
import quacc.evaluation.method as method

DATASET = "imdb"
OUTPUT_FILE = "out_" + DATASET + ".html"
TARGETS = {
    "rcv1" : [ 
        'C12', 
        'C13', 'C15', 'C151', 'C1511', 'C152', 'C17', 'C172', 
        'C18', 'C181', 'C21', 'C24', 'C31', 'C42', 'CCAT'
        'E11', 'E12', 'E21', 'E211', 'E212', 'E41', 'E51',  'ECAT',
        'G15', 'GCAT', 'GCRIM', 'GDIP', 'GPOL', 'GVIO', 'GVOTE', 'GWEA',
        'GWELF', 'M11', 'M12', 'M13', 'M131', 'M132', 'M14', 'M141',
        'M142', 'M143', 'MCAT'
    ],
    "spambase": ["default"],
    "imdb": ["default"],
}

def estimate_comparison():
    open(OUTPUT_FILE, "w").close()
    targets = TARGETS[DATASET]
    for target in targets:
        try:
            er = method.evaluate_comparison(DATASET, target=target)
            er.target = target
            with open(OUTPUT_FILE, "a") as f:
                f.write(er.to_html(["acc"], ["f1"]))
        except Exception:
            traceback.print_exc()

    # print(df.to_latex(float_format="{:.4f}".format))
    # print(utils.avg_group_report(df).to_latex(float_format="{:.4f}".format))


def main():
    estimate_comparison()


if __name__ == "__main__":
    main()
