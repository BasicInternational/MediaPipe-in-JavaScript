import cv2 from '@mjyc/opencv.js';

/// Expose the cv class (e.g. use it in the browser console)
window.cv2 = cv2;

/** Converts a rotation matrix (R) to Euler Angles  */
export function rotation_matrix_to_euler_angles(R) {
    const sy = Math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);
    const singular = sy < 1e-6;

    let x, y, z;
    if (!singular) {
        x = Math.atan2(R[2][1], R[2][2]);
        y = Math.atan2(-R[2][0], sy);
        z = Math.atan2(R[1][0], R[0][0]);
    } else {
        x = Math.atan2(-R[1][2], R[1][1]);
        y = Math.atan2(-R[2][0], sy);
        z = 0;
    }

    x = (x * 180) / Math.PI;
    y = (y * 180) / Math.PI;
    z = (z * 180) / Math.PI;

    return [x, y, z];
}

//! ref: get_rotate
/** Get Face Rotation (pitch, yaw) given the face image and landmarks  */
export function getFaceRotation(landMarks, image) {
    const { width, height } = image.size();
    // const channels = image.channels();

    const face_3d = [
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108],
        ],
        face_2d = [];

    landMarks.forEach((lm, idx) => {
        if ([1, 9, 57, 130, 287, 359].includes(idx)) {
            const [x, y] = [Math.floor(lm[0] * width), Math.floor(lm[1] * height)];

            /// Get the 2D Coordinates
            face_2d.push([x, y]);
        }
    });


    /// The camera matrix
    const focal_length = 1 * width;
    const center = [width / 2, height / 2];
    const cameraMatrix = cv2.matFromArray(3, 3, cv2.CV_32F, [
        focal_length,
        0,
        center[0],
        0,
        focal_length,
        center[1],
        0,
        0,
        1,
    ]);

    const distCoeffs = cv2.Mat.zeros(4, 1, cv2.CV_32F); // Assuming no lens distortion

    const face_3d_mat = cv2.matFromArray(6, 3, cv2.CV_32F, face_3d.flat());
    const face_2d_mat = cv2.matFromArray(6, 2, cv2.CV_32F, face_2d.flat());

    /** These matrices would be filled by `solvePnP`  */
    const rvec = new cv2.Mat({ width: 1, height: 3 }, cv2.CV_32F); /// Rotation Vector
    const tvec = new cv2.Mat({ width: 1, height: 3 }, cv2.CV_32F);

    cv2.solvePnP(
        face_3d_mat,
        face_2d_mat,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        true
    );


    /// Get rotational matrix
    const r = rodrigues([Array.from(rvec.data32F)]); /// data32F is used cause the matrix type if CV_32F

    /// Find the pitch yaw roll and check direction
    const [pitch, yaw] = rotation_matrix_to_euler_angles(r);

    return [pitch, yaw];
}

//! ref: check_landmarks_open
export function isMouthOpen(landmarks, image) {
    const { rows, cols } = image;

    const up = [
        Math.floor(landmarks[234][0] * cols),
        Math.floor(landmarks[234][1] * rows),
    ];

    const down = [
        Math.floor(landmarks[61][0] * cols),
        Math.floor(landmarks[61][1] * rows),
    ];

    const norm = distance(up, down);

    const mouth_up = [
        Math.floor(landmarks[13][0] * cols),
        Math.floor(landmarks[13][1] * rows),
    ];
    const mouth_down = [
        Math.floor(landmarks[14][0] * cols),
        Math.floor(landmarks[14][1] * rows),
    ];

    const rule_mouth = wisely(
        (i) => i / norm,
        abs(distance(mouth_up, mouth_down))
    );

    const open_mouth = wisely((i) => i > 0.11000000000000003, rule_mouth);
    const isMouthOpen = bool(open_mouth);

    return isMouthOpen;
}

/**** Helpers ****/

/** Euclidean distance 2x2 */
export const distance = (p1, p2) => Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));

/** Applys a function for each element in a given matrix
 *
 * fun - apply function
 * item - they matrix (would come to a single value in future recursive executions)
 * index - index of current element (not useful until `item` is a single value)
 */
const wisely = (fun, item, ...index) => {
    if (Array.isArray(item))
        return item.map((i, idx) => wisely(fun, i, idx, ...index));
    else return fun(item, ...index);
};


const abs = (arr) => wisely(Math.abs, arr);
const bool = (arr) => wisely(Boolean, arr);

const zeros = (r, c) => Array(r).fill(c ? Array(c).fill(0) : 0);
const ones = (r, c) => Array(r).fill(c ? Array(c).fill(1) : 1);
const clip = (arr, min, max) =>
    wisely((i) => (i > max ? max : i < min ? min : i), arr);

/** Calculates a transpose matrix for a given matrix */
const transpose = (a) =>
    a[0].map((_, colIndex) => a.map((row) => row[colIndex]));

/** Generates a `n by n` matrix where only the main diagonal is 1 other cells are 0  - Imitates numpy.eye  */
const eye = (n) =>
    Array(n)
        .fill(Array(n).fill(0))
        .map((a, i) => a.map((_, j) => (i === j ? 1 : 0)));

/** Dot multiply a number to a matrix */
const dotNumber = (n, a) => wisely((i) => i * n, a);
/** Element-wise multiplication two matrices */
const mMultiply = (a, b) => wisely((i, y, x) => i * b?.[x]?.[y] ?? 1, a);
/** Element-wise addition two matrices */
const mAdd = (a, b) => wisely((i, y, x) => i + b[x][y], a);

/** Impliments rodrigues algorithm to convert a given rotation vectors to rotation metrix  */
const rodrigues = (v) => {
    const [a, b, c] = v[0];

    const theta = Math.sqrt(Math.pow(a, 2) + Math.pow(b, 2) + Math.pow(c, 2));

    v = dotNumber(1 / theta, v);

    const v_ = v;
    const v_T = transpose(v);

    const m1 = dotNumber(Math.cos(theta), eye(3));
    const m2 = multiplyMatrices(v_T, v_);
    const m3 = dotNumber(1 - Math.cos(theta), m2);
    const m4 = dotNumber(Math.sin(theta), [
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ]);

    const r1 = mAdd(m1, m2);
    const r2 = mAdd(r1, m3);
    const r3 = mAdd(r2, m4);

    const R = r3;
    return R;
};

/// from: https://www.tutorialspoint.com/algorithm-for-matrix-multiplication-in-javascript
const multiplyMatrices = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        throw new Error('arguments should be in 2-dimensional array format');
    }
    let x = a.length,
        z = a[0].length,
        y = b[0].length;
    if (b.length !== z) {
        // XxZ & ZxY => XxY
        throw new Error(
            'number of columns in the first matrix should be the same as the number of rows in the second'
        );
    }
    let productRow = Array.apply(null, new Array(y)).map(
        Number.prototype.valueOf,
        0
    );
    let product = new Array(x);
    for (let p = 0; p < x; p++) {
        product[p] = productRow.slice();
    }
    for (let i = 0; i < x; i++) {
        for (let j = 0; j < y; j++) {
            for (let k = 0; k < z; k++) {
                product[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return product;
};
