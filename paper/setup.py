import matplotlib as mpl

# Figure sizes in inch
one_col = 3.37
two_col = 6.69 
std_height = one_col/1.62

def setup_matplotlib():
    fontsize = 8

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        #"font.family": "serif",
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern"],
        #"font.serif": [],                   # blank entries should cause plots 
        #"font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": fontsize,               # LaTeX default is 10pt font.
        "font.size": fontsize,
        "legend.fontsize": fontsize,               # Make the legend/label fonts 
        "xtick.labelsize": fontsize,               # a little smaller
        "ytick.labelsize": fontsize,
        "figure.figsize": (one_col,std_height),     # default fig size of 0.9 textwidth
        "figure.dpi": 300,
        "lines.linewidth": 1,
        "pgf.preamble": "\n".join([ # plots will use this preamble
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            #r"\usepackage[detect-all,locale=DE]{siunitx}",
            ])
        
    }
    mpl.rcParams.update(pgf_with_latex)

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'])  #https://personal.sron.nl/~pault/
    
    