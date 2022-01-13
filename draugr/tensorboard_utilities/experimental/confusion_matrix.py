#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

          If a tensorboard plugin exist when you are reading this, go use that :)
          
          TODO: Finish

           Created on 31-03-2021
           """

if __name__ == "__main__":

    def alt():
        """

        :return:
        """
        import tensorflow as tf
        import numpy

        import textwrap
        import re
        import io
        import itertools
        import matplotlib

        class SaverHook(tf.train.SessionRunHook):
            """
            Saves a confusion matrix as a Summary so that it can be shown in tensorboard
            """

            def __init__(self, labels, confusion_matrix_tensor_name, summary_writer):
                """Initializes a `SaveConfusionMatrixHook`.

                :param labels: Iterable of String containing the labels to print for each
                               row/column in the confusion matrix.
                :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
                                                     matrix
                :param summary_writer: The summary writer that will save the summary
                """
                self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
                self.labels = labels
                self._summary_writer = summary_writer

            def end(self, session):
                """

                :param session:
                """
                cm = (
                    tf.get_default_graph()
                    .get_tensor_by_name(self.confusion_matrix_tensor_name + ":0")
                    .eval(session=session)
                    .astype(int)
                )
                global_step = tf.train.get_global_step().eval(session=session)
                figure = self._plot_confusion_matrix(cm)
                summary = self._figure_to_summary(figure)
                self._summary_writer.add_summary(summary, global_step)

            def _figure_to_summary(self, fig):
                """
                Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
                that can be directly fed into ``Summary.FileWriter``.
                :param fig: A ``matplotlib.figure.Figure`` object.
                :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                         as a image summary.
                """

                # attach a new canvas if not exists
                if fig.canvas is None:
                    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()

                # get PNG data from the figure
                png_buffer = io.BytesIO()
                fig.canvas.print_png(png_buffer)
                png_encoded = png_buffer.getvalue()
                png_buffer.close()

                summary_image = tf.Summary.Image(
                    height=h,
                    width=w,
                    colorspace=4,  # RGB-A
                    encoded_image_string=png_encoded,
                )
                summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag=self.confusion_matrix_tensor_name, image=summary_image
                        )
                    ]
                )
                return summary

            def _plot_confusion_matrix(self, cm):
                """
                    :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
                `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
                """
                numClasses = len(self.labels)

                fig = matplotlib.figure.Figure(
                    figsize=(numClasses, numClasses),
                    dpi=100,
                    facecolor="w",
                    edgecolor="k",
                )
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(cm, cmap="Oranges")

                classes = [
                    re.sub(r"([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))", r"\1 ", x)
                    for x in self.labels
                ]
                classes = ["\n".join(textwrap.wrap(l, 20)) for l in classes]

                tick_marks = numpy.arange(len(classes))

                ax.set_xlabel("Predicted")
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(classes, rotation=-90, ha="center")
                ax.xaxis.set_label_position("bottom")
                ax.xaxis.tick_bottom()

                ax.set_ylabel("True Label")
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(classes, va="center")
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()

                for i, j in itertools.product(range(numClasses), range(numClasses)):
                    ax.text(
                        j,
                        i,
                        int(cm[i, j]) if cm[i, j] != 0 else ".",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                    )
                fig.set_tight_layout(True)
                return fig

    def asda():
        """

        :return:
        """
        import io
        import itertools
        from packaging import version

        import tensorflow as tf

        from matplotlib import pyplot
        import numpy

        test_pred_raw = []
        test_labels = []
        class_names = []
        train_labels = []
        train_images = []

        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            pyplot.savefig(buf, format="png")
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            pyplot.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def image_grid():
            """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
            # Create a figure to contain the plot.
            figure = pyplot.figure(figsize=(10, 10))
            for i in range(25):
                # Start next subplot.
                pyplot.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
                pyplot.xticks([])
                pyplot.yticks([])
                pyplot.grid(False)
                pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)

            return figure

        def plot_confusion_matrix(cm, class_names):
            """
            Returns a matplotlib figure containing the plotted confusion matrix.

            Args:
              cm (array, shape = [n, n]): a confusion matrix of integer classes
              class_names (array, shape = [n]): String names of the integer classes
            """
            figure = pyplot.figure(figsize=(8, 8))
            pyplot.imshow(cm, interpolation="nearest", cmap=pyplot.cm.Blues)
            pyplot.title("Confusion matrix")
            pyplot.colorbar()
            tick_marks = numpy.arange(len(class_names))
            pyplot.xticks(tick_marks, class_names, rotation=45)
            pyplot.yticks(tick_marks, class_names)

            # Compute the labels from the normalized confusion matrix.
            labels = numpy.around(
                cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis], decimals=2
            )

            # Use white text if squares are dark; otherwise black.
            threshold = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                pyplot.text(
                    j, i, labels[i, j], horizontalalignment="center", color=color
                )

            pyplot.tight_layout()
            pyplot.ylabel("True label")
            pyplot.xlabel("Predicted label")
            return figure

        """


test_pred = numpy.argmax(test_pred_raw, axis=1)

# Calculate the confusion matrix.
cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
# Log the confusion matrix as an image summary.
figure = plot_confusion_matrix(cm, class_names=class_names)
cm_image = plot_to_image(figure)

# Log the confusion matrix as an image summary.
with file_writer_cm.as_default():
tf.summary.image("Confusion Matrix", cm_image, step=epoch)
"""
