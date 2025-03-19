# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from collections import OrderedDict

from .modality import Modality


class ContextModality(Modality):
    """feature module
    Parameters
    ----------
    data: List[tuple], required
        A triplet list of user, item, and features, \
        e.g., data=[('user1', 'item1', ['feature1', 'feature2', 'feature3',...])].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', OrderedDict())

    @property
    def context(self):
        return self.__context

    @context.setter
    def context(self, input_context):
        self.__context = input_context

    @property
    def num_features(self):
        return len(self.feature_id_map)

    @property
    def user_context(self):
        return self.__user_context

    @user_context.setter
    def user_context(self, input_user_context):
        self.__user_context = input_user_context

    @property
    def item_context(self):
        return self.__item_context

    @item_context.setter
    def item_context(self, input_item_context):
        self.__item_context = input_item_context

    @property
    def feature_id_map(self):
        return self.__feature_id_map

    @feature_id_map.setter
    def feature_id_map(self, input_feature_id_map):
        self.__feature_id_map = input_feature_id_map

    def _build_context(self, uid_map, iid_map, dok_matrix):
        self.user_context = OrderedDict()
        self.item_context = OrderedDict()
        fid_map = OrderedDict()
        context = OrderedDict()
        for idx, (raw_uid, raw_iid, features) in enumerate(self.raw_data):
            user_idx = uid_map.get(raw_uid, None)
            item_idx = iid_map.get(raw_iid, None)
            if user_idx is None or item_idx is None or dok_matrix[user_idx, item_idx] == 0:
                continue
            user_dict = self.user_context.setdefault(user_idx, OrderedDict())
            user_dict[item_idx] = idx
            item_dict = self.item_context.setdefault(item_idx, OrderedDict())
            item_dict[user_idx] = idx

            mapped_features = []
            for feature in features:
                feature_idx = fid_map.setdefault(feature, len(fid_map))
                mapped_features.append(feature_idx)
            context.setdefault(idx, mapped_features)

        self.context = context
        self.feature_id_map = fid_map

    def build(self, uid_map=None, iid_map=None, dok_matrix=None, **kwargs):
        """Build the model based on provided list of ordered ids
        """
        if uid_map is not None and iid_map is not None and dok_matrix is not None:
            self._build_context(uid_map, iid_map, dok_matrix)
        return self
