# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class VideoData(KaitaiStruct):

    class MacroblockBType(Enum):
        b_direct_16x16 = 0
        b_l0_16x16 = 1
        b_l1_16x16 = 2
        b_bi_16x16 = 3
        b_l0_l0_16x8 = 4
        b_l0_l0_8x16 = 5
        b_l1_l1_16x8 = 6
        b_l1_l1_8x16 = 7
        b_l0_l1_16x8 = 8
        b_l0_l1_8x16 = 9
        b_l1_l0_16x8 = 10
        b_l1_l0_8x16 = 11
        b_l0_bi_16x8 = 12
        b_l0_bi_8x16 = 13
        b_l1_bi_16x8 = 14
        b_l1_bi_8x16 = 15
        b_bi_l0_16x8 = 16
        b_bi_l0_8x16 = 17
        b_bi_l1_16x8 = 18
        b_bi_l1_8x16 = 19
        b_bi_bi_16x8 = 20
        b_bi_bi_8x16 = 21
        b_8x8 = 22
        si = 23
        i_16x16_0_0_0 = 24
        i_16x16_1_0_0 = 25
        i_16x16_2_0_0 = 26
        i_16x16_3_0_0 = 27
        i_16x16_0_1_0 = 28
        i_16x16_1_1_0 = 29
        i_16x16_2_1_0 = 30
        i_16x16_3_1_0 = 31
        i_16x16_0_2_0 = 32
        i_16x16_1_2_0 = 33
        i_16x16_2_2_0 = 34
        i_16x16_3_2_0 = 35
        i_16x16_0_0_1 = 36
        i_16x16_1_0_1 = 37
        i_16x16_2_0_1 = 38
        i_16x16_3_0_1 = 39
        i_16x16_0_1_1 = 40
        i_16x16_1_1_1 = 41
        i_16x16_2_1_1 = 42
        i_16x16_3_1_1 = 43
        i_16x16_0_2_1 = 44
        i_16x16_1_2_1 = 46
        i_16x16_2_2_1 = 47
        i_16x16_3_2_1 = 48
        i_pcm = 49
        i_unknown = 50
        i_4x4 = 51
        i_8x8 = 52
        b_skip = 53

    class MacroblockIType(Enum):
        si = 0
        i_16x16_0_0_0 = 1
        i_16x16_1_0_0 = 2
        i_16x16_2_0_0 = 3
        i_16x16_3_0_0 = 4
        i_16x16_0_1_0 = 5
        i_16x16_1_1_0 = 6
        i_16x16_2_1_0 = 7
        i_16x16_3_1_0 = 8
        i_16x16_0_2_0 = 9
        i_16x16_1_2_0 = 10
        i_16x16_2_2_0 = 11
        i_16x16_3_2_0 = 12
        i_16x16_0_0_1 = 13
        i_16x16_1_0_1 = 14
        i_16x16_2_0_1 = 15
        i_16x16_3_0_1 = 16
        i_16x16_0_1_1 = 17
        i_16x16_1_1_1 = 18
        i_16x16_2_1_1 = 19
        i_16x16_3_1_1 = 20
        i_16x16_0_2_1 = 21
        i_16x16_1_2_1 = 22
        i_16x16_2_2_1 = 23
        i_16x16_3_2_1 = 24
        i_pcm = 25
        i_unknown = 26
        i_4x4 = 48
        i_8x8 = 49

    class MacroblockPType(Enum):
        p_l0_16x16 = 0
        p_l0_l0_16x8 = 1
        p_l0_l0_8x16 = 2
        p_8x8 = 3
        p_8x8ref0 = 4
        si = 5
        i_16x16_0_0_0 = 6
        i_16x16_1_0_0 = 7
        i_16x16_2_0_0 = 8
        i_16x16_3_0_0 = 9
        i_16x16_0_1_0 = 10
        i_16x16_1_1_0 = 11
        i_16x16_2_1_0 = 12
        i_16x16_3_1_0 = 13
        i_16x16_0_2_0 = 14
        i_16x16_1_2_0 = 15
        i_16x16_2_2_0 = 16
        i_16x16_3_2_0 = 17
        i_16x16_0_0_1 = 18
        i_16x16_1_0_1 = 19
        i_16x16_2_0_1 = 20
        i_16x16_3_0_1 = 21
        i_16x16_0_1_1 = 22
        i_16x16_1_1_1 = 23
        i_16x16_2_1_1 = 24
        i_16x16_3_1_1 = 25
        i_16x16_0_2_1 = 26
        i_16x16_1_2_1 = 27
        i_16x16_2_2_1 = 28
        i_16x16_3_2_1 = 29
        i_pcm = 30
        i_unknown = 31
        i_4x4 = 48
        i_8x8 = 49
        p_skip = 50

    class PlaneType(Enum):
        chroma_blue = 66
        luma = 76
        chroma_red = 82

    class SliceType(Enum):
        b_slice = 66
        i_slice = 73
        si_slice = 76
        p_slice = 80
        sp_slice = 81

    class StructureType(Enum):
        bottom_field = 66
        frame = 70
        top_field = 84
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.pictures = []
        i = 0
        while not self._io.is_eof():
            self.pictures.append(VideoData.PictureFrame(self._io, self, self._root))
            i += 1


    class MotionVector(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.list = self._io.read_u1()
            self.ref_idx = self._io.read_u1()
            self.diff_x = self._io.read_s2le()
            self.diff_y = self._io.read_s2le()
            self.abs_x = self._io.read_s2le()
            self.abs_y = self._io.read_s2le()


    class Macroblock(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.num = self._io.read_u4le()
            self.x_coo = self._io.read_u4le()
            self.y_coo = self._io.read_u4le()
            self.qp_y = self._io.read_u1()
            self.qp_u = self._io.read_s1()
            self.qp_v = self._io.read_s1()
            if self._parent.slice_type == VideoData.SliceType.i_slice:
                self.mb_i_type = KaitaiStream.resolve_enum(VideoData.MacroblockIType, self._io.read_u1())

            if self._parent.slice_type == VideoData.SliceType.p_slice:
                self.mb_p_type = KaitaiStream.resolve_enum(VideoData.MacroblockPType, self._io.read_u1())

            if self._parent.slice_type == VideoData.SliceType.b_slice:
                self.mb_b_type = KaitaiStream.resolve_enum(VideoData.MacroblockBType, self._io.read_u1())

            self.num_motion_vectors = self._io.read_u2le()
            self.motion_vectors = []
            for i in range(self.num_motion_vectors):
                self.motion_vectors.append(VideoData.MotionVector(self._io, self, self._root))

            self.dct_coefficients = []
            for i in range(3):
                self.dct_coefficients.append(VideoData.DctCoeff(self._io, self, self._root))



    class Slice(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.num = self._io.read_u2le()
            self.slice_type = KaitaiStream.resolve_enum(VideoData.SliceType, self._io.read_u1())
            self.num_macroblocks = self._io.read_u4le()
            self.macroblocks = []
            for i in range(self.num_macroblocks):
                self.macroblocks.append(VideoData.Macroblock(self._io, self, self._root))



    class DctCoeff(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.plane = KaitaiStream.resolve_enum(VideoData.PlaneType, self._io.read_u1())
            self.num_values = self._io.read_u2le()
            self.values = []
            for i in range(self.num_values):
                self.values.append(self._io.read_s4le())



    class SubPicture(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.structure = KaitaiStream.resolve_enum(VideoData.StructureType, self._io.read_u1())
            self.num_slices = self._io.read_u2le()
            self.slices = []
            for i in range(self.num_slices):
                self.slices.append(VideoData.Slice(self._io, self, self._root))



    class PictureFrame(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.picture_id = self._io.read_u4le()
            self.poc = self._io.read_u4le()
            self.gop_num = self._io.read_u4le()
            self.num_sub_pictures = self._io.read_u2le()
            self.sub_pictures = []
            for i in range(self.num_sub_pictures):
                self.sub_pictures.append(VideoData.SubPicture(self._io, self, self._root))




