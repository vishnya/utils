import seaborn as sns
import matplotlib.pyplot as plt


def plot_barplots_with_hue(df, hue, categorical_cols):
    for col in categorical_cols:
        sns.countplot(x=col, hue=hue, data=df)
        plt.show()


def set_sns_settings():
    sns.set(style="darkgrid", palette="muted", color_codes=True)


def rotate_plot_x_ticks(degrees):
    """
    Alternatively, e.g.:
    f = sns.lineplot(x=ctr_ts.index, y=ctr_ts.click)
    f.tick_params(labelrotation=45)
    :param degrees:
    :return:
    """
    plt.xticks(rotation=degrees)


def create_four_subplots():
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text('First Plot')
    ax2.title.set_text('Second Plot')
    ax3.title.set_text('Third Plot')
    ax4.title.set_text('Fourth Plot')
    plt.show()
    """
    pass


def format_plot(f, rotation, title, xlabel, ylabel):
    f.tick_params(labelrotation=rotation)
    f.set_title(title)
    f.set_xlabel(xlabel)
    f.set_ylabel(ylabel)
