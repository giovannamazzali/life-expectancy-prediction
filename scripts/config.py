import matplotlib.pyplot as plt
import matplotlib.colors as clrs

from cycler import cycler




SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




plt.style.use('dslabs.mplstyle')

my_palette = {'yellow': '#ECD474', 'pale orange': '#E9AE4E', 'salmon': '#E2A36B', 'orange': '#F79522', 'dark orange': '#D7725E',
              'pale acqua': '#92C4AF', 'acqua': '#64B29E', 'marine': '#3D9EA9', 'green': '#10A48A', 'olive': '#99C244',
              'pale blue': '#BDDDE0', 'blue2': '#199ED5', 'blue3': '#1DAFE5', 'dark blue': '#0C70B2',
              'pale pink': '#D077AC', 'pink': '#EA4799', 'lavender': '#E09FD5', 'lilac': '#B081B9', 'purple': '#923E97',
              'white': '#FFFFFF', 'light grey': '#D2D3D4', 'grey': '#939598', 'black': '#000000'}

colors_pale = [my_palette['salmon'], my_palette['blue2'], my_palette['acqua']]
colors_soft = [my_palette['dark orange'], my_palette['dark blue'], my_palette['green']]
colors_live = [my_palette['orange'], my_palette['blue3'], my_palette['olive']]
blues = [my_palette['pale blue'], my_palette['blue2'], my_palette['blue3'], my_palette['dark blue']]
oranges = [my_palette['pale orange'], my_palette['salmon'], my_palette['orange'], my_palette['dark orange']]

LINE_COLOR = my_palette['dark blue']
FILL_COLOR = my_palette['pale blue']
DOT_COLOR = my_palette['blue3']
ACTIVE_COLORS = [my_palette['dark blue'], my_palette['yellow'], my_palette['pale orange'],
                 my_palette['acqua'], my_palette['pale pink'], my_palette['lavender']]
cmap_orange = clrs.LinearSegmentedColormap.from_list("myCMPOrange", oranges)
cmap_blues = clrs.LinearSegmentedColormap.from_list("myCMPBlues", blues)
cmap_active = clrs.LinearSegmentedColormap.from_list("myCMPActive", ACTIVE_COLORS)

alpha = 0.3
plt.rcParams['axes.prop_cycle'] = cycler('color', ACTIVE_COLORS)

plt.rcParams['text.color'] = LINE_COLOR
plt.rcParams['patch.edgecolor'] = LINE_COLOR
plt.rcParams['patch.facecolor'] = FILL_COLOR
plt.rcParams['axes.facecolor'] = my_palette['white']
plt.rcParams['axes.edgecolor'] = my_palette['grey']
plt.rcParams['axes.labelcolor'] = my_palette['grey']
plt.rcParams['xtick.color'] = my_palette['grey']
plt.rcParams['ytick.color'] = my_palette['grey']

plt.rcParams['grid.color'] = my_palette['light grey']

plt.rcParams['boxplot.boxprops.color'] = FILL_COLOR
plt.rcParams['boxplot.capprops.color'] = LINE_COLOR
plt.rcParams['boxplot.flierprops.color'] = my_palette['pink']
plt.rcParams['boxplot.flierprops.markeredgecolor'] = FILL_COLOR
plt.rcParams['boxplot.flierprops.markerfacecolor'] = FILL_COLOR
plt.rcParams['boxplot.whiskerprops.color'] = LINE_COLOR
plt.rcParams['boxplot.meanprops.color'] = my_palette['purple']
plt.rcParams['boxplot.meanprops.markeredgecolor'] = my_palette['purple']
plt.rcParams['boxplot.meanprops.markerfacecolor'] = my_palette['purple']
plt.rcParams['boxplot.medianprops.color'] = my_palette['green']