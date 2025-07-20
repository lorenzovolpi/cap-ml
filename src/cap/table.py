import os
import pdb
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats, wilcoxon


@dataclass
class Format:
    mean_prec: int = 3
    std_prec: int = 3
    meanrank_prec: int = 1
    meanrank_std_prec: int = 1
    show_std: bool = True
    remove_zero: bool = False
    color: bool = True
    maxtone: int = 40
    style: str = "minimal"
    lower_is_better: bool = True
    stat_test: str = "wilcoxon"
    show_stat: bool = True
    simple_stat: bool = False
    color_mode: str = "local"
    with_mean: bool = True
    mean_macro: bool = True
    with_rank_mean: bool = True
    only_full_mean: bool = True
    best_color: str = "green"
    mid_color: str = "cyan"
    worst_color: str = "red"
    baseline_color: str = "yellow"


class Cell:
    def __init__(
        self,
        format: Format,
        local_group: "CellGroup",
        global_group: "CellGroup",
        baseline_group: "CellGroup",
        is_baseline: bool = False,
    ):
        self.values = []
        self.format = format
        self.touch()
        self.local_group = local_group
        self.local_group.register_cell(self)
        self.global_group = global_group
        self.global_group.register_cell(self)
        self.baseline_group = baseline_group
        if is_baseline:
            self.baseline_group.register_cell(self)

    def __len__(self):
        return len(self.values)

    def mean(self):
        if self.mean_ is None:
            self.mean_ = np.mean(self.values)
        return self.mean_

    def rank(self, global_scope=False) -> int:
        group = self.global_group if global_scope else self.local_group
        return group.rank(self)

    def std(self):
        if self.std_ is None:
            self.std_ = np.std(self.values)
        return self.std_

    def touch(self):
        self.mean_ = None
        self.std_ = None

    def append(self, v: Union[float, Iterable]):
        if isinstance(v, Iterable):
            self.values.extend(v)
        else:
            self.values.append(v)
        self.touch()

    def isEmpty(self):
        return len(self) == 0

    def isBest(self, global_scope=False):
        group = self.global_group if global_scope else self.local_group
        best = group.best()
        if best is not None:
            return (best == self) or (np.isclose(best.mean(), self.mean()))
        return False

    def print_mean(self):
        if self.isEmpty():
            return ""
        else:
            return f"{self.mean():.{self.format.mean_prec}f}"

    def print(self):
        if self.isEmpty():
            return ""

        whitespace = "$^{\phantom{\dag}}$" if self.format.show_stat else ""

        # mean
        # ---------------------------------------------------
        mean = self.print_mean()
        if self.format.remove_zero:
            mean = mean.replace("0.", ".")

        # std ?
        # ---------------------------------------------------
        if self.format.show_std:
            _std_val = f"{self.std():.{self.format.std_prec}f}"
            if self.format.remove_zero:
                _std_val = _std_val.replace("0.", ".")
            std = f"$\pm${whitespace}{_std_val}"
        else:
            std = ""

        # bold or statistical test
        # ---------------------------------------------------
        if self.isBest():
            str_cell = f"\\textbf{{{mean}{whitespace}{std}}}"
        else:
            comp_symbol = whitespace
            pval = self.local_group.compare(self)
            if pval is not None and self.format.simple_stat:
                if 0.001 > pval:
                    comp_symbol = whitespace
                else:
                    comp_symbol = "$^{\dag}$"
            elif pval is not None and not self.format.simple_stat:
                if 0.001 > pval:
                    comp_symbol = whitespace
                elif 0.05 > pval >= 0.001:
                    comp_symbol = "$^{\dag}$"
                elif pval >= 0.05:
                    comp_symbol = "$^{\ddag}$"
            str_cell = f"{mean}{comp_symbol}{std}"

        str_cell = str_cell.replace("$$", "")  # remove useless "$$"

        # color ?
        # ---------------------------------------------------
        if self.format.color:
            group = {
                "local": self.local_group,
                "global": self.global_group,
                "baselines": self.baseline_group,
            }[self.format.color_mode]
            # str_cell += ' ' + group.color(self)
            str_cell += group.color(self)

        return str_cell


class CellGroup:
    def __init__(self, format: Format):
        assert format.stat_test in [
            "wilcoxon",
            "ttest",
            None,
        ], f"unknown {format.stat_test=}, valid ones are wilcoxon, ttest, or None"
        assert format.color_mode in [
            "local",
            "global",
            "baselines",
        ], f"unknown {format.color_mode=}, valid ones are local and global"
        # if (format.color_global_min is not None or format.color_global_max is not None) and format.color_mode=='local':
        #     print('warning: color_global_min and color_global_max are only considered when color_mode==local')
        self.format = format
        self.cells = []

    def register_cell(self, cell: Cell):
        self.cells.append(cell)

    def non_empty_cells(self):
        return [c for c in self.cells if not c.isEmpty()]

    def max(self):
        cells = self.non_empty_cells()
        if len(cells) > 0:
            return cells[np.argmax([c.mean() for c in cells])]
        return None

    def min(self):
        cells = self.non_empty_cells()
        if len(cells) > 0:
            return cells[np.argmin([c.mean() for c in cells])]
        return None

    def best(self) -> Cell:
        return self.min() if self.format.lower_is_better else self.max()

    def worst(self) -> Cell:
        return self.max() if self.format.lower_is_better else self.min()

    def isEmpty(self):
        return len(self.non_empty_cells()) == 0

    def compare(self, cell: Cell):
        best = self.best()
        best_n = len(best)
        cell_n = len(cell)
        if best_n > 0 and cell_n > 0:
            if self.format.stat_test == "wilcoxon":
                try:
                    _, p_val = wilcoxon(best.values, cell.values)
                except ValueError:
                    p_val = None
                return p_val
            elif self.format.stat_test == "ttest":
                best_mean, best_std = best.mean(), best.std()
                cell_mean, cell_std = cell.mean(), cell.std()
                _, p_val = ttest_ind_from_stats(best_mean, best_std, best_n, cell_mean, cell_std, cell_n)
                return p_val
            elif self.format.stat_test is None:
                return None
            else:
                raise ValueError(f"unknown statistical test {self.stat_test}")
        else:
            return None

    def color(self, cell: Cell):
        if self.format.color_mode == "baselines":
            assert not self.isEmpty(), "Invalid value for format.color_mode: no baselines found"
            best_bline = self.best()

            if cell in self.non_empty_cells():
                if cell.mean() == best_bline.mean():
                    color = self.format.baseline_color
                    tone = self.format.maxtone
                    return f"\cellcolor{{{color}!{int(tone)}}}"
                else:
                    return ""

            if self.format.lower_is_better:
                cell_is_better = cell.mean() <= best_bline.mean()
            else:
                cell_is_better = cell.mean() >= best_bline.mean()

            p_val = self.compare(cell)

            # color = self.format.best_color if cell_is_better else self.format.worst_color
            # if p_val < 0.001:
            #     tone = self.format.maxtone
            # elif p_val >= 0.001 and p_val < 0.05:
            #     tone = self.format.maxtone * 0.7
            # else:
            #     tone = self.format.maxtone * 0.5
            # return f"\cellcolor{{{color}!{int(tone)}}}"

            if cell_is_better:
                tone = self.format.maxtone
                if p_val < 0.001:
                    color = self.format.best_color
                else:
                    # tone = self.format.maxtone * 0.6
                    color = self.format.mid_color
                return f"\cellcolor{{{color}!{int(tone)}}}"
            else:
                return ""

        else:
            cell_mean = cell.mean()

            # if self.format.color_mode == 'local':
            best = self.best()
            worst = self.worst()
            best_mean = best.mean()
            worst_mean = worst.mean()

            if best is None or worst is None or best_mean == worst_mean or cell.isEmpty():
                return ""

            # normalize val in [0,1]
            maxval = max(best_mean, worst_mean)
            minval = min(best_mean, worst_mean)
            # else:
            #     maxval = self.format.color_global_max
            #     minval = self.format.color_global_min

            normval = (cell_mean - minval) / (maxval - minval)

            if self.format.lower_is_better:
                normval = 1 - normval

            normval = np.clip(normval, 0, 1)

            normval = normval * 2 - 1  # rescale to [-1,1]
            if normval < 0:
                color = cell.format.worst_color
                tone = cell.format.maxtone * (-normval)
            else:
                color = cell.format.best_color
                tone = cell.format.maxtone * normval

            return f"\cellcolor{{{color}!{int(tone)}}}"

    def rank(self, cell: Cell) -> int:
        if cell.isEmpty():
            return None
        cell_mean = cell.mean()
        all_means = [c.mean() for c in self.cells]
        # all_means.append(cell_mean)
        all_means = set(all_means)  # remove duplicates so that ties receive the same rank
        all_means = [x for x in all_means if not np.isnan(x)]  # remove nan
        all_means = sorted(all_means)
        return all_means.index(cell_mean) + 1


class Table:
    def __init__(
        self,
        name="table",
        benchmarks=None,
        methods=None,
        baselines=None,
    ):
        self.name = name
        self.benchmarks = [] if benchmarks is None else benchmarks
        self.methods = [] if methods is None else methods
        self.baselines = [] if baselines is None else baselines
        self.format = Format()

        # if self.format.color_mode == 'global':
        #     self.format.color_global_min = 0
        #     self.format.color_global_max = 1
        # else:
        #     self.format.color_global_min = None
        #     self.format.color_global_max = None

        self.T = {}
        self.bline_groups = {}
        self.groups = {}
        self.global_group = self._new_group()
        self.left_frame = None

    def add(self, benchmark, method, v):
        cell = self.get(benchmark, method)
        cell.append(v)

    def set_left_frame(self, label: str):
        self.left_frame = label

    def get_benchmarks(self):
        return self.benchmarks

    def get_methods(self):
        return self.methods

    def n_benchmarks(self):
        return len(self.benchmarks)

    def n_methods(self):
        return len(self.methods)

    def _new_group(self):
        return CellGroup(self.format)

    def get(self, benchmark, method) -> Cell:
        if benchmark not in self.benchmarks:
            self.benchmarks.append(benchmark)
        if benchmark not in self.groups:
            self.groups[benchmark] = self._new_group()
        if benchmark not in self.bline_groups:
            self.bline_groups[benchmark] = self._new_group()
        if method not in self.methods:
            self.methods.append(method)
        b_idx = self.benchmarks.index(benchmark)
        m_idx = self.methods.index(method)
        idx = tuple((b_idx, m_idx))
        if idx not in self.T:
            self.T[idx] = Cell(
                self.format,
                local_group=self.groups[benchmark],
                global_group=self.global_group,
                baseline_group=self.bline_groups[benchmark],
                is_baseline=method in self.baselines,
            )
        cell = self.T[idx]
        return cell

    def get_mean_value(self, benchmark, method) -> float:
        cell = self.get(benchmark, method)
        return cell.mean()

    def get_benchmark_cells(self, benchmark):
        cells = [self.get(benchmark=benchmark, method=m) for m in self.get_methods()]
        cells = [c for c in cells if not c.isEmpty()]
        return cells

    def get_method_cells(self, method):
        cells = [self.get(benchmark=b, method=method) for b in self.get_benchmarks()]
        cells = [c for c in cells if not c.isEmpty()]
        return cells

    def get_method_means(self, method_order):
        mean_group = self._new_group()
        mean_bline_group = self._new_group()
        mean_global_group = self._new_group()
        cells = []
        for method in method_order:
            is_baseline = method in self.baselines
            method_mean = Cell(
                self.format,
                local_group=mean_group,
                global_group=mean_global_group,
                baseline_group=mean_bline_group,
                is_baseline=is_baseline,
            )
            leave_empty = False
            for bench in self.get_benchmarks():
                if self.format.mean_macro:
                    # macro: adds the mean value for bench and method (mean across the means)
                    mean_value = self.get_mean_value(benchmark=bench, method=method)
                    add_value = not np.isnan(mean_value)
                else:
                    # micro: adds all values for bench and method (means across all values)
                    mean_value = self.get(benchmark=bench, method=method).values
                    add_value = mean_value is not None

                if add_value:
                    method_mean.append(mean_value)
                elif self.format.only_full_mean:
                    leave_empty = True

                if leave_empty:
                    break  # with only one missing value, the average should not be computed

            if leave_empty:
                cells.append(
                    Cell(
                        self.format,
                        local_group=mean_group,
                        global_group=mean_global_group,
                        baseline_group=mean_bline_group,
                        is_baseline=is_baseline,
                    )
                )
            else:
                cells.append(method_mean)

        return cells

    def get_method_rank_means(self, method_order):
        rank_mean_group = self._new_group()
        rank_mean_bline_group = self._new_group()
        rank_global_group = self._new_group()
        cells = []
        for method in method_order:
            is_baseline = method in self.baselines
            rankmean_format = replace(self.format)
            rankmean_format.mean_prec = self.format.meanrank_prec
            rankmean_format.std_prec = self.format.meanrank_std_prec
            method_rank_mean = Cell(
                format=rankmean_format,
                local_group=rank_mean_group,
                global_group=rank_global_group,
                baseline_group=rank_mean_bline_group,
                is_baseline=is_baseline,
            )
            leave_empty = False
            for bench in self.get_benchmarks():
                # if bench == r"\textsf{iris.3}" and method == r"PrediQuant":
                #     pdb.set_trace()
                rank = self.get(benchmark=bench, method=method).rank()
                if rank is not None:
                    method_rank_mean.append(rank)
                elif self.format.only_full_mean:
                    leave_empty = True
                if leave_empty:
                    break
            if leave_empty:
                cells.append(
                    Cell(
                        self.format,
                        local_group=rank_mean_group,
                        global_group=rank_global_group,
                        baseline_group=rank_mean_bline_group,
                        is_baseline=is_baseline,
                    )
                )
            else:
                cells.append(method_rank_mean)
        return cells

    def get_benchmark_values(self, benchmark):
        values = np.asarray([c.mean() for c in self.get_benchmark_cells(benchmark)])
        return values

    def get_method_values(self, method):
        values = np.asarray([c.mean() for c in self.get_method_cells(method)])
        return values

    def all_mean(self):
        values = [c.mean() for c in self.T.values() if not c.isEmpty()]
        return np.mean(values)

    def print(self):  # todo: missing method names?
        data_dict = {}
        data_dict["Benchmark"] = [b for b in self.get_benchmarks()]
        for method in self.get_methods():
            data_dict[method] = [self.get(bench, method).print_mean() for bench in self.get_benchmarks()]
        df = pd.DataFrame(data_dict)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        print(df.to_string(index=False))

    def tabular(
        self,
        path=None,
        benchmark_replace=None,
        method_replace=None,
        benchmark_order=None,
        method_order=None,
        transpose=False,
    ):
        if benchmark_replace is None:
            benchmark_replace = {}
        if method_replace is None:
            method_replace = {}
        if benchmark_order is None:
            benchmark_order = self.get_benchmarks()
        if method_order is None:
            method_order = self.get_methods()

        if transpose:
            row_order, row_replace = method_order, method_replace
            col_order, col_replace = benchmark_order, benchmark_replace
        else:
            row_order, row_replace = benchmark_order, benchmark_replace
            col_order, col_replace = method_order, method_replace

        n_cols = len(col_order)
        add_mean_col = self.format.with_mean and transpose
        add_mean_row = self.format.with_mean and not transpose
        last_col_idx = n_cols + 2 if add_mean_col else n_cols + 1
        average_label = "\\textit{Average}"
        rank_average_label = "\\textit{Rank}"

        if self.format.with_mean:
            mean_cells = self.get_method_means(method_order)
            rankmean_cells = self.get_method_rank_means(method_order)

        lines = []
        if self.format.style == "full":
            toprule = f"\\cline{{2-{last_col_idx}}}"
            endl = " \\\\\\hline"
            midrule = endl
            bottomrule = endl
            corner = "\multicolumn{1}{c|}{} & "

            begin_tabular = "\\begin{tabular}{|c" + "|c" * n_cols + ("||c" if add_mean_col else "") + "|}"
        elif self.format.style == "rules":
            toprule = "\\toprule"
            endl = " \\\\"
            midrule = " \\\\\\midrule"
            bottomrule = " \\\\\\bottomrule"
            corner = " & "

            begin_tabular = "\\begin{tabular}{c" + "c" * n_cols + ("c" if add_mean_col else "") + "}"
        elif self.format.style == "minimal":
            toprule = f"\\cline{{2-{last_col_idx}}}"
            endl = " \\\\"
            midrule = " \\\\\\hline"
            bottomrule = " \\\\\\hline"
            corner = "\multicolumn{1}{c|}{} & "

            begin_tabular = "\\begin{tabular}{|c|" + "c" * n_cols + ("|c" if add_mean_col else "") + "|}"
        else:
            raise ValueError(f"unknown format style {self.format.style}")

        lines.append(begin_tabular)
        lines.append(toprule)

        l = corner
        l += " & ".join([col_replace.get(col, col) for col in col_order])
        if add_mean_col:
            l += " & " + average_label
        l += midrule
        lines.append(l)

        # printed_table = [[self.get(benchmark=col if transpose else row, method=row if transpose else col).print() for col in col_order] for row in row_order]
        # printed_lengths = np.zeros(len(row_order), len(col_order))
        # for i in range(len(row_order)):
        #     for j in range(len(col_order)):
        #         printed_lengths[i,j] = len(printed_table[i][j])
        # col_width = printed_lengths.max(axis=0)

        for i, row in enumerate(row_order):
            rowname = row_replace.get(row, row)
            l = rowname + " & "
            l += " & ".join(
                [
                    self.get(benchmark=col if transpose else row, method=row if transpose else col).print()
                    for col in col_order
                ]
            )
            if add_mean_col:
                l += " & " + mean_cells[i].print()

            if i < len(row_order) - 1:
                l += endl
            else:  # last line
                if add_mean_row:  # midrule, since there will be an additional row
                    l += midrule
                else:
                    l += bottomrule  # bottomrule, this is indeed the last row
            lines.append(l)

        if add_mean_row:
            l = average_label + " & "
            l += " & ".join([mean_cell.print() for mean_cell in mean_cells])

            if self.format.with_rank_mean:
                l += "\\\\"
                lines.append(l)
                l = rank_average_label + " & "
                l += " & ".join([mean_cell.print() for mean_cell in rankmean_cells])

            l += bottomrule
            lines.append(l)

        lines.append("\\end{tabular}")

        tabular_tex = "\n".join(lines)

        if path is not None:
            parent = Path(path).parent
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "wt") as foo:
                foo.write(tabular_tex)

        return tabular_tex

    def table(
        self,
        tabular_path,
        benchmark_replace=None,
        method_replace=None,
        resizebox=True,
        caption=None,
        label=None,
        benchmark_order=None,
        method_order=None,
        transpose=False,
    ):
        if benchmark_replace is None:
            benchmark_replace = {}
        if method_replace is None:
            method_replace = {}

        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\center")
        if resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")

        tabular_str = self.tabular(
            tabular_path, benchmark_replace, method_replace, benchmark_order, method_order, transpose
        )
        if tabular_path is None:
            lines.append(tabular_str)
        else:
            lines.append(f"\input{{tables/{Path(tabular_path).name}}}")

        if resizebox:
            lines.append("}%")
        if caption is None:
            caption = tabular_path.replace("_", "\_")
        lines.append(f"\caption{{{caption}}}")
        if label is not None:
            lines.append(f"\label{{{label}}}")
        lines.append("\end{table}")

        table_tex = "\n".join(lines)

        return table_tex

    def document(self, tex_path, tabular_dir="tables", *args, **kwargs):
        Table.Document(tex_path, tables=[self], tabular_dir=tabular_dir, *args, **kwargs)

    def latexPDF(self, pdf_path, tabular_dir="tables", *args, **kwargs):
        return Table.LatexPDF(pdf_path, tables=[self], tabular_dir=tabular_dir, *args, **kwargs)

    @classmethod
    def Document(
        self,
        tex_path,
        tables: List["Table"],
        tabular_dir="tables",
        landscape=True,
        dedicated_pages=True,
        *args,
        **kwargs,
    ):
        lines = []
        lines.append("\\documentclass[10pt,a4paper]{article}")
        lines.append("\\usepackage[utf8]{inputenc}")
        lines.append("\\usepackage{amsmath}")
        lines.append("\\usepackage{amsfonts}")
        lines.append("\\usepackage{amssymb}")
        lines.append("\\usepackage{graphicx}")
        lines.append("\\usepackage[dvipsnames]{xcolor}")
        lines.append("\\usepackage{colortbl}")
        lines.append("\\usepackage{booktabs}")
        if landscape:
            lines.append("\\usepackage[landscape]{geometry}")
        lines.extend(kwargs.pop("new_commands", []))
        lines.append("")
        lines.append("\\begin{document}")
        for table in tables:
            lines.append("")
            lines.append(
                table.table(
                    os.path.join(Path(tex_path).parent, tabular_dir, table.name + "_table.tex"), *args, **kwargs
                )
            )
            lines.append("\n")
            if dedicated_pages:
                lines.append("\\newpage\n")
        lines.append("\\end{document}")

        document = "\n".join(lines)

        parent = Path(tex_path).parent
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tex_path, "wt") as foo:
            foo.write(document)

        return document

    @classmethod
    def LatexPDF(cls, pdf_path: str, tables: List["Table"], tabular_dir: str = "tables", *args, **kwargs):
        assert pdf_path.endswith(".pdf"), f"{pdf_path=} does not seem a valid name for a pdf file"
        tex_path = pdf_path.replace(".pdf", ".tex")

        cls.Document(tex_path, tables, tabular_dir, *args, **kwargs)

        dir = Path(pdf_path).parent
        pwd = os.getcwd()

        print("[Tables Done] runing latex")
        os.chdir(dir)
        tex_path_name = Path(tex_path).name
        command = ["pdflatex", tex_path_name]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        basename = tex_path_name.replace(".tex", "")
        os.system(f"rm {basename}.aux {basename}.log {basename}.tex")
        os.chdir(pwd)
        print("[Done]")
