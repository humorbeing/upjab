import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111)

color_gt = 'black'
# color_ce_curve = '#941504'
color_ce_curve = 'red'
color_ce_points = '#1a3a4a'
color_ce_area = '#f9d5c1'
color_test_curve = '#005a8c'
color_test_area = '#cbe3e0'


def ce_curve(x1, y1, x2, y2, ax, color='gray', label=None):

    def interp(xs, ys, num_points=10000, x_start=0, x_end=100):
        x_cut = np.linspace(x_start, x_end, num=num_points, endpoint=True)    
        y_points = np.interp(x_cut, xs, ys)    
        return x_cut, y_points


    x1, y1 = interp(x1, y1)
    x2, y2 = interp(x2, y2)

    def area_between(xc1, yp1, xc2, yp2):
        from shapely.geometry import Polygon
        x_y_curve1 = []

        for i, j in zip(xc1, yp1):
            x_y_curve1.append((i,j))

        x_y_curve2 = []

        for i, j in zip(xc2, yp2):
            x_y_curve2.append((i,j))

        polygon_points = [] #creates a empty list where we will append the points to create the polygon

        for xyvalue in x_y_curve1:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

        for xyvalue in x_y_curve2[::-1]:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

        for xyvalue in x_y_curve1[0:1]:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

        polygon = Polygon(polygon_points)
        area = polygon.area
        return area

    area_ = area_between(x1, y1, x2, y2)

    ax.fill(
        np.append(x1, x2[::-1]),
        np.append(y1, y2[::-1]),
        facecolor=color,
        # alpha=0.4,
        label=label
    )

    return area_



def draw_ce(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    datas = lines[1:]

    def get_value(s_):
        sp = s_.strip().split(' ')
        temp11 = sp[0]
        temp12 = sp[-1]
        propor = float(temp11.split('->')[-1][:-1])
        remain = 1- float(temp12)
        
        return propor, remain
        
    xs = []
    ys = []
    for d_ in datas:
        propor, remain = get_value(d_)
        xs.append(remain*100)
        ys.append(propor*100)
        # print('')
    xs.insert(0, 0)
    ys.insert(0, 0)
    xs.append(100)
    ys.append(10)
    return xs, ys


def ce_point_estimation(histogram_group_count, proportions):

    histogram_group_count = np.array(histogram_group_count)
    proportions = np.array(proportions)
    total_n = histogram_group_count.sum()
    ab_list = []
    nor_list = []
    for histo_, propor_ in zip(histogram_group_count, proportions):    
        abn = histo_*propor_
        nn = histo_ - abn
        # use float to increase accuracy
        ab_list.append(abn)
        nor_list.append(nn)
        
    ab_list = np.array(ab_list)
    nor_list = np.array(nor_list)
    ab_t = ab_list.sum()
    remain_all = total_n
    ab_left = ab_t

    x_axis = []
    y_axis = []
    for i in reversed(range(1, 10)):
        remain_all = remain_all - histogram_group_count[i]
        ab_left = ab_left - ab_list[i]
        x_is = remain_all / total_n
        pp = ab_left / remain_all
        x_axis.append(x_is*100)
        y_axis.append(pp*100)
        # print()

    x_ce = x_axis
    y_ce = y_axis

    x_ce.reverse()
    y_ce.reverse()

    x_ce.insert(0, 0)
    y_ce.insert(0, 0)
    x_ce.append(100)
    y_ce.append(10)

    return x_ce, y_ce

histogram_group_count=[
    9786,
    654,327,264,179,
    146,128,107,113,
    285
]


proportions = [
    0.04,
    0.105875, 0.17174999999999999, 0.23762499999999998, 0.30349999999999994, 0.36937499999999995, 0.4352499999999999, 0.5011249999999999,
    0.567, 0.9316
]

xs_ce, ys_ce = ce_point_estimation(histogram_group_count, proportions)

# Groud Truth
data_path = 'uu/plot_gallery/data/pre-histogram_final_pick_low_all_train_metric.txt'
xs_gt, ys_gt = draw_ce(data_path)


data_path = 'uu/plot_gallery/data/pre-histogram_final_pick_low_labeled_test_metric.txt'
xs_test, ys_test = draw_ce(data_path)



area1 = ce_curve(
    xs_gt,
    ys_gt,
    xs_test,
    ys_test,
    ax1,
    color_test_area,
    label='CE Error (10%)'
    )


ax1.plot(
    xs_test,
    ys_test,
    color=color_test_curve,
    label='10% Samples',
    linewidth=0.8
    )


area2 = ce_curve(
    xs_gt,
    ys_gt,
    xs_ce,
    ys_ce,
    ax1,
    color_ce_area,
    label='CE Error (Our)'
    )

ax1.plot(
    xs_gt,
    ys_gt,
    color=color_gt,
    label='Ground Truth',
    linewidth=1.3
    )



# Test Set




ax1.plot(
    xs_ce,
    ys_ce,
    label='Our',
    color=color_ce_curve,
    linewidth=1.1
    )






ax1.scatter(
    xs_ce,
    ys_ce,
    color=color_ce_points,
    marker='x',
    # s=50,
    label='Key Points',
    )



handles, labels = ax1.get_legend_handles_labels()
order = [3,1,0,5,4,2]


plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
ax1.grid(True)
plt.xlabel('% of Remaining Samples')
plt.ylabel('% of Abnormal Sample in Remaining Samples')


# plt.savefig('cleanse_effectiveness/ce_curve.pdf')
# plt.savefig('cleanse_effectiveness/ce_curve.png')
# plt.savefig('cleanse_effectiveness/ce_curve.svg')
plt.show()




print()