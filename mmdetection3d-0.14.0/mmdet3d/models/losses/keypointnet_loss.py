import torch
import math
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class KeypointnetLoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0):
        super(KeypointnetLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                z,
                target,
                image_size,
                num_kp,
                images,
                dimension,
                transformer,
                orient=None):

        # Get variance loss parameters
        vh = image_size[0]
        vw = image_size[1]
        ranx, rany = self.meshgrid(vh, vw)
        prob_variance = prob.reshape(-1, num_kp, vh, vw)
        sx = torch.sum(prob_variance * ranx, dim=[2, 3])
        sy = torch.sum(prob_variance * rany, dim=[2, 3])
        uv = torch.stack([sx, sy], dim=-1).reshape(-1, num_kp, 2)
        z = torch.sum(prob * z, dim=[2, 3])

        mask = images
        mask = (mask > torch.zeros_like(mask)).float()
        sill = torch.sum(prob * mask.unsqueeze(1), dim=[2, 3])
        sill = torch.mean(-torch.log(sill + 1e-12))

        uv = uv.view(-1, num_kp, 2)
        z = z.view(-1, num_kp, 1)

        #loss_lr = tf.losses.mean_squared_error(orient[:, :, :2], xp[:, :, :2])
        loss_variance = self.variance_loss(prob_variance, ranx, rany, uv)
        loss_sill = sill
        loss_consistency = None
        loss_separation = None
        loss_chordal = None
        loss_angular = None

        return loss_variance, loss_sill, loss_consistency, loss_separation, loss_chordal, loss_angular

    @staticmethod
    def meshgrid(h, w):
        """Returns a meshgrid ranging from [-1, 1] in x, y axes."""
        r_x = (torch.arange(0.5, w, 1) / (w / 2)) - 1
        r_y = (torch.arange(0.5, h, 1) / (h / 2)) - 1
        ranx, rany = torch.meshgrid(r_x, -r_y, indexing="xy")
        return ranx.float(), rany.float()

    @staticmethod
    def estimate_rotation(xyz0, xyz1, pconf, noise):
        """Estimates the rotation between two sets of keypoints.

        The rotation is estimated by first subtracting mean from each set of keypoints
        and computing SVD of the covariance matrix.

        Args:
        xyz0: [batch, num_kp, 3] The first set of keypoints.
        xyz1: [batch, num_kp, 3] The second set of keypoints.
        pconf: [batch, num_kp] The weights used to compute the rotation estimate.
        noise: A number indicating the noise added to the keypoints.

        Returns:
        [batch, 3, 3] A batch of transposed 3 x 3 rotation matrices.
        """

        # Add random noise with normal distribution
        xyz0 += torch.randn_like(xyz0) * noise
        xyz1 += torch.randn_like(xyz1) * noise

        # Expand pconf along the last dimension
        pconf2 = pconf.unsqueeze(2)

        # Compute weighted centers
        cen0 = torch.sum(xyz0 * pconf2, dim=1, keepdim=True)
        cen1 = torch.sum(xyz1 * pconf2, dim=1, keepdim=True)

        # Center the coordinates
        x = xyz0 - cen0
        y = xyz1 - cen1

        # Compute covariance matrix
        cov = torch.matmul(torch.matmul(x.transpose(1, 2), torch.diag_embed(pconf)), y)

        # Perform SVD on the covariance matrix
        u, s, v = torch.svd(cov)

        # Compute the determinant of the matrix multiplication of v and u.T
        d = torch.det(torch.matmul(v, u.transpose(1, 2)))

        # Adjust u using the determinant
        ud = torch.cat([u[:, :, :-1], u[:, :, -1:] * d.unsqueeze(1).unsqueeze(1)], dim=2)

        # Return the product of ud and v.T
        return torch.matmul(ud, v.transpose(1, 2))

        # Original TensorFlow code:
        # xyz0 += tf.random_normal(tf.shape(xyz0), mean=0, stddev=noise)
        # xyz1 += tf.random_normal(tf.shape(xyz1), mean=0, stddev=noise)
        #
        # pconf2 = tf.expand_dims(pconf, 2)
        # cen0 = tf.reduce_sum(xyz0 * pconf2, 1, keepdims=True)
        # cen1 = tf.reduce_sum(xyz1 * pconf2, 1, keepdims=True)
        #
        # x = xyz0 - cen0
        # y = xyz1 - cen1
        #
        # cov = tf.matmul(tf.matmul(x, tf.matrix_diag(pconf), transpose_a=True), y)
        # _, u, v = tf.svd(cov, full_matrices=True)
        #
        # d = tf.matrix_determinant(tf.matmul(v, u, transpose_b=True))
        # ud = tf.concat(
        #   [u[:, :, :-1], u[:, :, -1:] * tf.expand_dims(tf.expand_dims(d, 1), 1)],
        #   axis=2)
        # return tf.matmul(ud, v, transpose_b=True)

    def relative_pose_loss(self, xyz0, xyz1, rot, pconf, noise):
        """Computes the relative pose loss (chordal, angular).

        Args:
        xyz0: [batch, num_kp, 3] The first set of keypoints.
        xyz1: [batch, num_kp, 3] The second set of keypoints.
        rot: [batch, 4, 4] The ground-truth rotation matrices.
        pconf: [batch, num_kp] The weights used to compute the rotation estimate.
        noise: A number indicating the noise added to the keypoints.

        Returns:
        A tuple (chordal loss, angular loss).
        """

        r_transposed = self.estimate_rotation(xyz0, xyz1, pconf, noise)
        rotation = rot[:, :3, :3]

        # Frobenius norm squared (element-wise square difference, sum over last two dimensions)
        frob_sqr = torch.sum((r_transposed - rotation) ** 2, dim=[1, 2])

        # Frobenius norm (square root of frob_sqr)
        frob = torch.sqrt(frob_sqr)

        # First return value: Mean of frobenius squared
        mean_frob_sqr = torch.mean(frob_sqr)

        # Second return value: 2 * mean of asin(min(1.0, frob / (2 * sqrt(2))))
        asin_term = torch.asin(torch.clamp(frob / (2 * math.sqrt(2)), max=1.0))
        mean_asin_term = 2.0 * torch.mean(asin_term)

        return mean_frob_sqr, mean_asin_term

        # Original TensorFlow code:
        # r_transposed = estimate_rotation(xyz0, xyz1, pconf, noise)
        # rotation = rot[:, :3, :3]
        # frob_sqr = tf.reduce_sum(tf.square(r_transposed - rotation), axis=[1, 2])
        # frob = tf.sqrt(frob_sqr)
        #
        # return tf.reduce_mean(frob_sqr), \
        #   2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / (2 * math.sqrt(2)))))

    @staticmethod
    def separation_loss(xyz, delta):
        """Computes the separation loss.

        Args:
        xyz: [batch, num_kp, 3] Input keypoints.
        delta: A separation threshold. Incur 0 cost if the distance >= delta.

        Returns:
        The seperation loss.
        """

        num_kp = xyz.shape[1]

        batch_size = xyz.shape[0]

        # Repeat xyz along the second dimension
        t1 = xyz.repeat(1, num_kp, 1)

        # Repeat xyz along both first and second dimensions, then reshape
        t2 = xyz.repeat(1, 1, num_kp).view_as(t1)

        # Compute the squared difference
        diffsq = (t1 - t2) ** 2

        # Sum over the last dimension -> [batch, num_kp ^ 2]
        lensqr = torch.sum(diffsq, dim=2)

        # Apply max operation to (-lensqr + delta) and 0.0
        result = torch.sum(torch.clamp(-lensqr + delta, min=0.0))

        # Calculate the final result
        result = result / (num_kp * batch_size * 2.0)

        return result

        # Original TensorFlow code:
        # num_kp = tf.shape(xyz)[1]
        # t1 = tf.tile(xyz, [1, num_kp, 1])
        #
        # t2 = tf.reshape(tf.tile(xyz, [1, 1, num_kp]), tf.shape(t1))
        # diffsq = tf.square(t1 - t2)
        #
        # # -> [batch, num_kp ^ 2]
        # lensqr = tf.reduce_sum(diffsq, axis=2)
        #
        # return (tf.reduce_sum(tf.maximum(-lensqr + delta, 0.0)) / tf.to_float(
        #   num_kp * FLAGS.batch_size * 2))

    @staticmethod
    def consistency_loss(uv0, uv1, pconf):
        """Computes multi-view consistency loss between two sets of keypoints.

        Args:
            uv0: [batch, num_kp, 2] The first set of keypoint 2D coordinates.
            uv1: [batch, num_kp, 2] The second set of keypoint 2D coordinates.
            pconf: [batch, num_kp] The weights used to compute the rotation estimate.

        Returns:
            The consistency loss.
        """

        # [batch, num_kp, 2]
        wd = (uv0 - uv1) ** 2 * pconf.unsqueeze(2)
        wd = torch.sum(wd, dim=[1, 2])
        return torch.mean(wd)

        # Original TensorFlow code:
        # # [batch, num_kp, 2]
        # wd = tf.square(uv0 - uv1) * tf.expand_dims(pconf, 2)
        # wd = tf.reduce_sum(wd, axis=[1, 2])
        # return tf.reduce_mean(wd)

    @staticmethod
    def variance_loss(probmap, ranx, rany, uv):
        """Computes the variance loss as part of Sillhouette consistency.

        Args:
            probmap: [batch, num_kp, h, w] The distribution map of keypoint locations.
            ranx: X-axis meshgrid.
            rany: Y-axis meshgrid.
            uv: [batch, num_kp, 2] Keypoint locations (in NDC).

        Returns:
            The variance loss.
        """

        # Stack ranx and rany along dimension 2
        ran = torch.stack([ranx, rany], dim=2)

        # Get shape of ran
        sh = ran.shape
        # Reshape ran to [1, 1, sh[0], sh[1], 2]
        ran = ran.view(1, 1, sh[0], sh[1], 2)

        # Get shape of uv
        sh = uv.shape
        # Reshape uv to [sh[0], sh[1], 1, 1, 2]
        uv = uv.view(sh[0], sh[1], 1, 1, 2)

        # Calculate the squared difference and reduce sum over axis 4
        diff = torch.sum((uv - ran) ** 2, dim=4)
        # Multiply by probmap
        diff *= probmap

        # Reduce sum over dimensions 2 and 3, then take the mean
        result = torch.mean(torch.sum(diff, dim=[2, 3]))

        return result

        # Original TensorFlow code:
        # ran = tf.stack([ranx, rany], axis=2)
        #
        # sh = tf.shape(ran)
        # # [batch, num_kp, vh, vw, 2]
        # ran = tf.reshape(ran, [1, 1, sh[0], sh[1], 2])
        #
        # sh = tf.shape(uv)
        # uv = tf.reshape(uv, [sh[0], sh[1], 1, 1, 2])
        #
        # diff = tf.reduce_sum(tf.square(uv - ran), axis=4)
        # diff *= probmap
        #
        # return tf.reduce_mean(tf.reduce_sum(diff, axis=[2, 3]))

