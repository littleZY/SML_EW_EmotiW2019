import numpy as np
import pandas as pd
import os

class EngagementDataset():
    def __init__(self, openface_features, vggface2_features, au_features, engagement_labels, n_segments=15, alpha=0.5,
                 augment=0, shuffle=False):
        """
        Initialization
        :param openface_features:
        :param engagement_labels:
        :param n_segments:
        :param alpha:
        """
        self.n_segments = n_segments
        self.alpha = alpha
        self.random_state = 97

        labels_df = pd.read_csv(engagement_labels, header=None).values
        self.vd_names = labels_df[:, 0]
        self.vd_lbls = labels_df[:, 1].astype(float)
        self.lbl_dict = {}
        for idx, name in enumerate(self.vd_names):
            self.lbl_dict[name] = self.vd_lbls[idx]

        ld_with_ext = os.listdir(openface_features)
        ld_no_ext = [x[:-4] for x in ld_with_ext if x.endswith('.csv')]

        self.ld_no_ext = []
        self.ld_csv = []
        self.ld_vggface2 = []
        self.ld_au = []
        self.ld_dfscore = []

        for x in ld_no_ext:
            if x in self.vd_names:
                self.ld_csv.append(os.path.join(openface_features, x + '.csv'))
                self.ld_vggface2.append(os.path.join(vggface2_features, x + '.npz'))
                self.ld_au.append(os.path.join(au_features, x + '.npz'))
                self.ld_no_ext.append(x)
                self.ld_dfscore.append(self.lbl_dict[x])

        self.ld_csv = np.array(self.ld_csv)
        self.ld_vggface2 = np.array(self.ld_vggface2)
        self.ld_au = np.array(self.ld_au)
        self.ld_no_ext = np.array(self.ld_no_ext)
        self.ld_dfscore = np.array(self.ld_dfscore)

        if augment:
            minor_idx = np.where(self.ld_dfscore == 0.0)[0]
            augment_idx = skutils.resample(minor_idx, random_state=self.random_state, replace=True,
                                           n_samples=augment - len(minor_idx))
            new_idx = np.hstack([np.arange(self.ld_dfscore.shape[0]), augment_idx])
        else:
            new_idx = np.arange(self.ld_dfscore.shape[0])

        # new_idx = new_idx[:5]
        if shuffle:
            new_idx = skutils.shuffle(new_idx, random_state=self.random_state + 2)

        self.ld_csv = self.ld_csv[new_idx]
        self.ld_vggface2 = self.ld_vggface2[new_idx]
        self.ld_au = self.ld_au[new_idx]
        self.ld_no_ext = self.ld_no_ext[new_idx]
        self.ld_dfscore = self.ld_dfscore[new_idx]

    def get_gaze_features(self, raw_input):
        """
        Get gaze features from raw input
        :param raw_input:
        :return:
        """
        # Get statiscal feature from raw input
        gaze_direction = raw_input[:, 5:11]
        gaze_angle = raw_input[:, 11: 13]
        eye_landmark2D = raw_input[:, 13: 125]
        eye_landmark3D = raw_input[:, 125: 293]
        pose_direction = raw_input[:, 293: 299]
        face_landmark2D = raw_input[:, 299: 435]
        face_landmark3D = raw_input[:, 435: 679]
        au_reg = raw_input[:, 679: 695]
        au_cls = raw_input[:, 695: 713]

        gaze_direction_std = np.std(gaze_direction, axis=0)
        gaze_direction_mean = np.mean(gaze_direction, axis=0)

        gaze_angle_std = np.std(gaze_angle, axis=0)
        gaze_angle_mean = np.mean(gaze_angle, axis=0)

        eye_landmark2D_shape_0 = np.abs(eye_landmark2D[:, 56 + 9: 56 + 14] - eye_landmark2D[:, 56 + 19: 56 + 14: -1])
        eye_landmark2D_shape_1 = np.abs(eye_landmark2D[:, 56 + 37: 56 + 42] - eye_landmark2D[:, 56 + 47: 56 + 42: -1])
        eye_landmark2D_shape = np.hstack((eye_landmark2D_shape_0, eye_landmark2D_shape_1))
        eye_landmark2D_shape_cov = np.divide(np.std(eye_landmark2D_shape, axis=0),
                                             np.mean(eye_landmark2D_shape, axis=0))

        eye_distance = 0.5 * (eye_landmark3D[:, 56 * 2 + 8] + eye_landmark3D[:, 56 * 2 + 42])
        eye_distance_cov = np.std(eye_distance) / np.mean(eye_distance)
        eye_distance_ratio = np.min(eye_distance) / np.max(eye_distance)
        eye_distance_fea = np.array([eye_distance_cov, eye_distance_ratio])

        eye_location2D = []
        for idx in range(4):
            cur_mean = np.mean(eye_landmark2D[:, 28 * idx: 28 * (idx + 1)], axis=1)
            eye_location2D.append(cur_mean)

        eye_location2D = np.vstack(eye_location2D).T
        eye_location2D_mean = np.mean(eye_location2D, axis=0)
        eye_location2D_std = np.std(eye_location2D, axis=0)

        eye_location3D = []
        for idx in range(6):
            cur_mean = np.mean(eye_landmark3D[:, 28 * idx: 28 * (idx + 1)], axis=1)
            eye_location3D.append(cur_mean)
        eye_location3D = np.vstack(eye_location3D).T
        eye_location3D_mean = np.mean(eye_location3D, axis=0)
        eye_location3D_std = np.std(eye_location3D, axis=0)

        pose_direction_mean = np.mean(pose_direction, axis=0)
        pose_direction_std = np.std(pose_direction, axis=0)
        ret_features = np.hstack((gaze_direction_std, gaze_direction_mean, gaze_angle_mean, gaze_angle_std,
                                  eye_landmark2D_shape_cov, eye_location2D_mean, eye_location2D_std,
                                  eye_location3D_mean,
                                  eye_location3D_std, eye_distance_fea, pose_direction_mean, pose_direction_std))

        return ret_features

    def parse_gaze_features(self, txt_path):
        """
        Divide txt to n_segments with same size
        the end of the last segment: k + (n_segments-1)*(1-alpha)*k
        try to choose alpha and n_segments in order to (n_segments-1)*(1-alpha) is integer
        :param txt_path:
        :param n_segments:
        :param alpha: overlap percent
        :return:
        """
        df = pd.read_csv(txt_path, header=0, sep=',').values
        face_id = df[:, 1]
        seq_length = df.shape[0]

        indexing = int((self.n_segments - 1) * (1 - self.alpha))
        k_value = seq_length // (1 + indexing)  # In some case, we will ignore some last frames

        ret = []
        index_st = 0
        for idx in range(self.n_segments):
            index_ed = k_value + int(k_value * (1 - self.alpha) * idx)
            index_features = self.get_gaze_features(df[index_st: index_ed, :])
            ret.append(index_features)
            index_st = index_ed - int((1 - self.alpha) * k_value)

        ret = np.vstack(ret)
        return ret

    def get_au_features(self, raw_input, ft_type):
        """

        :param raw_input:
        :param ft_type: 0 - mean, 1 - std, 2 - max, -1 - all
        :return:
        """
        if ft_type == 0:
            ret = np.mean(raw_input, axis=0)
            ret = np.sum(ret, axis=0)
        elif ft_type == 1:
            ret = np.std(raw_input, axis=0)
            ret = np.sum(ret, axis=0)
        elif ft_type == 2:
            ret = np.max(raw_input, axis=0)
            ret = np.sum(ret, axis=0)
        else:
            ret0 = np.mean(raw_input, axis=0)
            ret1 = np.std(raw_input, axis=0)
            ret2 = np.max(raw_input, axis=0)
            ret = np.vstack([ret0, ret1, ret2])

        return ret.reshape(1, -1)

    def parse_au_features(self, au_path, sft=2):
        dauz = np.load(os.path.join(au_path), allow_pickle=True)['values']
        seq_length = dauz.shape[0]

        indexing = int((self.n_segments - 1) * (1 - self.alpha))
        k_value = seq_length // (1 + indexing)  # In some case, we will ignore some last frames

        ret = []
        index_st = 0
        for idx in range(self.n_segments):
            index_ed = k_value + int(k_value * (1 - self.alpha) * idx)
            # ft_type = [0, 1, 2, -1] - [mean, std, max, all]
            index_features = self.get_au_features(dauz[index_st: index_ed, :], ft_type=sft)
            ret.append(index_features)
            index_st = index_ed - int((1 - self.alpha) * k_value)

        ret = np.vstack(ret)
        return ret

    def get_vgg2_features(self, raw_input, ft_type):
        """

        :param raw_input:
        :param ft_type: 0 - mean, 1 - std, 2 - max, -1 - all
        :return:
        """
        if ft_type == 0:
            ret = np.mean(raw_input, axis=0).reshape(1, -1)
        elif ft_type == 1:
            ret = np.std(raw_input, axis=0).reshape(1, -1)
        elif ft_type == 2:
            ret = np.max(raw_input, axis=0).reshape(1, -1)
        else:
            ret0 = np.mean(raw_input, axis=0).reshape(1, -1)
            ret1 = np.std(raw_input, axis=0).reshape(1, -1)
            ret2 = np.max(raw_input, axis=0).reshape(1, -1)
            ret = np.vstack([ret0, ret1, ret2])

        return ret

    def parse_vgg2_features(self, vgg_path, sft=2):
        dvggz = np.load(os.path.join(vgg_path), allow_pickle=True)['values']
        seq_length = dvggz.shape[0]

        indexing = int((self.n_segments - 1) * (1 - self.alpha))
        k_value = seq_length // (1 + indexing)  # In some case, we will ignore some last frames

        ret = []
        index_st = 0
        for idx in range(self.n_segments):
            index_ed = k_value + int(k_value * (1 - self.alpha) * idx)
            # ft_type = [0, 1, 2, -1] - [mean, std, max, all]
            index_features = self.get_vgg2_features(dvggz[index_st: index_ed, :], ft_type=sft)
            ret.append(index_features)
            index_st = index_ed - int((1 - self.alpha) * k_value)

        ret = np.vstack(ret)
        return ret

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ld_csv)

    def get_item(self, idx, ft_type, sft=2):
        """
        Generate one sample of data
        :param idx:
        :param ft_type: [0, 1, 2] - [gaze, au, vgg]
        :param sft: 0, 1, 2, -1, feature type (mean, std, max, all) for gaze, au, vgg respectively
        :return:
        """
        txt_score = self.lbl_dict[self.ld_no_ext[idx]]
        if ft_type == 0:
            txt_name = self.ld_csv[idx]
            X = self.parse_gaze_features(txt_name)
        elif ft_type == 1:
            txt_name = self.ld_au[idx]
            X = self.parse_au_features(txt_name, sft)
        elif ft_type == 2:
            txt_name = self.ld_vggface2[idx]
            X = self.parse_vgg2_features(txt_name, sft)
        else:
            raise "Do not support ft_type = {}".format(ft_type)
        return X, txt_score

    def get_all_gaze_features(self):
        pass

    def get_all_au_features(self):
        pass

    def get_all_face_features(self):
        pass

    def get_all_data(self, ft_list, sft, num_jobs=4, shuffle=False, augment=False):
        """
        Get all data
        :param ft_list: list, [0, 1, 2] - [gaze, au, vgg]
        :param sft: list, [-1, 1, 1], feature type (mean, std, max, all) for gaze, au, vgg respectively
        :param num_jobs: num_job parallel
        :param shuffle:
        :param augment: augment minority class (0) by upsampling or not
        :return:
        """

        ret = []
        for ft in ft_list:
            print(ft)
            current_parallel = Parallel(n_jobs=num_jobs)(
                delayed(self.get_item)(ix, ft, sft[ft]) for ix in range(len(self.ld_csv)))

            ld_features = []
            ld_scores = []
            for tup in current_parallel:
                ld_features.append(tup[0])
                ld_scores.append(tup[1])

            ld_features = np.stack(ld_features)
            ld_scores = np.array(ld_scores)

            ret.append((ld_features, ld_scores))
        return ret
