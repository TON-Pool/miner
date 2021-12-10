// This file belongs to TON-Pool.com Miner (https://github.com/TON-Pool/miner)
// License: GPLv3

__kernel void hash_solver_1(__global const uint* args, __global uint* res) {
  uint idx = get_global_id(0);
  uint iterations = args[0];
  uint global_it = args[1];
  const uint txargs2 = args[2];
  const uint txargs3 = args[3];
  const uint txargs4 = args[4];
  const uint txargs5 = args[5];
  const uint txargs6 = args[6];
  const uint txargs7 = args[7];
  const uint txargs8 = args[8];
  const uint txargs9 = args[9];
  const uint txargs10 = args[10];
  const uint txargs11 = args[11];
  const uint txargs12 = args[12];
  const uint txargs13 = args[13];
  const uint txargs14 = args[14];
  const uint txargs15 = args[15];
  const uint txargs16 = args[16];
  const uint txargs17 = args[17];
  const uint txargs18 = args[18];
  const uint txargs19 = args[19];
  const uint txargs20 = args[20];
  const uint txargs21 = args[21];
  const uint txargs22 = args[22];
  for (uint i = 0; i < iterations; i++) {
    uint a = txargs2;
    uint b = txargs3;
    uint c = txargs4;
    uint d = txargs5;
    uint e = txargs6;
    uint f = txargs7;
    uint g = txargs8;
    uint h = txargs9;
    uint w0_t = txargs10 ^ i;
    uint w1_t = txargs11 ^ idx;
    uint w2_t = txargs12 ^ global_it;
    uint w3_t = txargs13;
    uint w4_t = txargs14;
    uint w5_t = txargs15;
    uint w6_t = txargs16;
    uint w7_t = txargs17;
    uint w8_t = txargs18;
    uint w9_t = txargs19;
    uint wa_t = txargs20;
    uint wb_t = txargs21;
    uint wc_t = txargs10 ^ i;
    uint wd_t = txargs11 ^ idx;
    uint we_t = txargs22 ^ global_it;
    uint wf_t = 0;
    sha256_transform();
    uint oa = a += txargs2;
    b += txargs3;
    c += txargs4;
    d += txargs5;
    e += txargs6;
    f += txargs7;
    g += txargs8;
    uint oh = h += txargs9;
    w0_t = 0;
    w1_t = 0;
    w2_t = 0;
    w3_t = 0;
    w4_t = 0;
    w5_t = 0;
    w6_t = 0;
    w7_t = 0;
    w8_t = 0;
    w9_t = 0;
    wa_t = 0;
    wb_t = 0;
    wc_t = 0;
    wd_t = 0;
    we_t = 0;
    wf_t = 984;
    sha256_transform();
    if (oa + a == 0) {
      uint pos = oh & 2046;
      res[pos] = idx;
      res[pos + 1] = i;
    }
  }
}

__kernel void hash_solver_2(__global const uint* args, __global uint* res) {
  uint idx = get_global_id(0);
  uint iterations = args[0];
  uint global_it = args[1];
  for (uint i = 0; i < iterations; i++) {
    uint a = args[2];
    uint b = args[3];
    uint c = args[4];
    uint d = args[5];
    uint e = args[6];
    uint f = args[7];
    uint g = args[8];
    uint h = args[9];
    uint w0_t = args[10] ^ i;
    uint w1_t = args[11] ^ idx;
    uint w2_t = args[12] ^ global_it;
    uint w3_t = args[13];
    uint w4_t = args[14];
    uint w5_t = args[15];
    uint w6_t = args[16];
    uint w7_t = args[17];
    uint w8_t = args[18];
    uint w9_t = args[19];
    uint wa_t = args[20];
    uint wb_t = args[21];
    uint wc_t = args[10] ^ i;
    uint wd_t = args[11] ^ idx;
    uint we_t = args[22] ^ global_it;
    uint wf_t = 0;
    sha256_transform();
    uint oa = a += args[2];
    b += args[3];
    c += args[4];
    d += args[5];
    e += args[6];
    f += args[7];
    g += args[8];
    uint oh = h += args[9];
    w0_t = 0;
    w1_t = 0;
    w2_t = 0;
    w3_t = 0;
    w4_t = 0;
    w5_t = 0;
    w6_t = 0;
    w7_t = 0;
    w8_t = 0;
    w9_t = 0;
    wa_t = 0;
    wb_t = 0;
    wc_t = 0;
    wd_t = 0;
    we_t = 0;
    wf_t = 984;
    sha256_transform();
    if (oa + a == 0) {
      uint pos = oh & 2046;
      res[pos] = idx;
      res[pos + 1] = i;
    }
  }
}

__kernel void hash_solver_3(__global const uint* args, __global uint* res) {
  uint idx = get_global_id(0);
  uint iterations = args[0];
  uint global_it = args[1];
  const uint txargs2 = args[2];
  const uint txargs3 = args[3];
  const uint txargs4 = args[4];
  const uint txargs5 = args[5];
  const uint txargs6 = args[6];
  const uint txargs7 = args[7];
  const uint txargs8 = args[8];
  const uint txargs9 = args[9];
  const uint txargs10 = args[10];
  const uint txargs11 = args[11];
  const uint txargs12 = args[12];
  const uint txargs13 = args[13];
  const uint txargs14 = args[14];
  const uint txargs15 = args[15];
  const uint txargs16 = args[16];
  const uint txargs17 = args[17];
  const uint txargs18 = args[18];
  const uint txargs19 = args[19];
  const uint txargs20 = args[20];
  const uint txargs21 = args[21];
  const uint txargs22 = args[22];
  for (uint i = 0; i < iterations; i++) {
    uint a = txargs2;
    uint b = txargs3;
    uint c = txargs4;
    uint d = txargs5;
    uint e = txargs6;
    uint f = txargs7;
    uint g = txargs8;
    uint h = txargs9;
    uint w0_t = txargs10 ^ i;
    uint w1_t = txargs11 ^ idx;
    uint w2_t = txargs12 ^ global_it;
    uint w3_t = txargs13;
    uint w4_t = txargs14;
    uint w5_t = txargs15;
    uint w6_t = txargs16;
    uint w7_t = txargs17;
    uint w8_t = txargs18;
    uint w9_t = txargs19;
    uint wa_t = txargs20;
    uint wb_t = txargs21;
    uint wc_t = txargs10 ^ i;
    uint wd_t = txargs11 ^ idx;
    uint we_t = txargs22 ^ global_it;
    uint wf_t = 0;
    sha256_transform();
    uint oa = a += args[2];
    b += args[3];
    c += args[4];
    d += args[5];
    e += args[6];
    f += args[7];
    g += args[8];
    uint oh = h += args[9];
    w0_t = 0;
    w1_t = 0;
    w2_t = 0;
    w3_t = 0;
    w4_t = 0;
    w5_t = 0;
    w6_t = 0;
    w7_t = 0;
    w8_t = 0;
    w9_t = 0;
    wa_t = 0;
    wb_t = 0;
    wc_t = 0;
    wd_t = 0;
    we_t = 0;
    wf_t = 984;
    sha256_transform();
    if (oa + a == 0) {
      uint pos = oh & 2046;
      res[pos] = idx;
      res[pos + 1] = i;
    }
  }
}

__kernel void hash_solver_4(__global const uint* args, __global uint* res) {
  uint idx = get_global_id(0);
  uint iterations = args[0];
  uint global_it = args[1];
  const uint txargs2 = args[2];
  const uint txargs3 = args[3];
  const uint txargs4 = args[4];
  const uint txargs5 = args[5];
  const uint txargs6 = args[6];
  const uint txargs7 = args[7];
  const uint txargs8 = args[8];
  const uint txargs9 = args[9];
  const uint txargs10 = args[10];
  const uint txargs11 = args[11];
  const uint txargs12 = args[12];
  const uint txargs13 = args[13];
  const uint txargs14 = args[14];
  const uint txargs15 = args[15];
  const uint txargs16 = args[16];
  const uint txargs17 = args[17];
  const uint txargs18 = args[18];
  const uint txargs19 = args[19];
  const uint txargs20 = args[20];
  const uint txargs21 = args[21];
  const uint txargs22 = args[22];
  for (uint i = 0; i < iterations; i++) {
    uint a = txargs2;
    uint b = txargs3;
    uint c = txargs4;
    uint d = txargs5;
    uint e = txargs6;
    uint f = txargs7;
    uint g = txargs8;
    uint h = txargs9;
    {
      uint w0_t = txargs10 ^ i;
      uint w1_t = txargs11 ^ idx;
      uint w2_t = txargs12 ^ global_it;
      uint w3_t = txargs13;
      uint w4_t = txargs14;
      uint w5_t = txargs15;
      uint w6_t = txargs16;
      uint w7_t = txargs17;
      uint w8_t = txargs18;
      uint w9_t = txargs19;
      uint wa_t = txargs20;
      uint wb_t = txargs21;
      uint wc_t = txargs10 ^ i;
      uint wd_t = txargs11 ^ idx;
      uint we_t = txargs22 ^ global_it;
      uint wf_t = 0;
      sha256_transform();
    }
    uint oa = a += txargs2;
    b += txargs3;
    c += txargs4;
    d += txargs5;
    e += txargs6;
    f += txargs7;
    g += txargs8;
    uint oh = h += txargs9;
    {
      const uint w0_t = 0;
      const uint w1_t = 0;
      const uint w2_t = 0;
      const uint w3_t = 0;
      const uint w4_t = 0;
      const uint w5_t = 0;
      const uint w6_t = 0;
      const uint w7_t = 0;
      const uint w8_t = 0;
      const uint w9_t = 0;
      const uint wa_t = 0;
      const uint wb_t = 0;
      const uint wc_t = 0;
      const uint wd_t = 0;
      const uint we_t = 0;
      const uint wf_t = 984;
      sha256_transform();
    }
    if (oa + a == 0) {
      uint pos = oh & 2046;
      res[pos] = idx;
      res[pos + 1] = i;
    }
  }
}