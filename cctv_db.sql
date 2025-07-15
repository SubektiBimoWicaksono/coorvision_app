-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 15 Jul 2025 pada 06.48
-- Versi server: 10.4.32-MariaDB
-- Versi PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cctv_db`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `alembic_version`
--

CREATE TABLE `alembic_version` (
  `version_num` varchar(32) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `alembic_version`
--

INSERT INTO `alembic_version` (`version_num`) VALUES
('98fa557179df');

-- --------------------------------------------------------

--
-- Struktur dari tabel `camera_settings`
--

CREATE TABLE `camera_settings` (
  `id` int(11) NOT NULL,
  `cam_name` varchar(200) NOT NULL,
  `feed_src` varchar(200) DEFAULT NULL,
  `x1` int(11) DEFAULT NULL,
  `y1` int(11) DEFAULT NULL,
  `x2` int(11) DEFAULT NULL,
  `y2` int(11) DEFAULT NULL,
  `x3` int(11) DEFAULT NULL,
  `y3` int(11) DEFAULT NULL,
  `x4` int(11) DEFAULT NULL,
  `y4` int(11) DEFAULT NULL,
  `x5` int(11) DEFAULT NULL,
  `y5` int(11) DEFAULT NULL,
  `x6` int(11) DEFAULT NULL,
  `y6` int(11) DEFAULT NULL,
  `x7` int(11) DEFAULT NULL,
  `y7` int(11) DEFAULT NULL,
  `x8` int(11) DEFAULT NULL,
  `y8` int(11) DEFAULT NULL,
  `cam_is_active` tinyint(1) DEFAULT NULL,
  `gender_detection` tinyint(1) DEFAULT NULL,
  `face_detection` tinyint(1) DEFAULT NULL,
  `face_capture` tinyint(1) DEFAULT NULL,
  `id_card_detection` tinyint(1) DEFAULT NULL,
  `uniform_detection` tinyint(1) DEFAULT NULL,
  `shoes_detection` tinyint(1) DEFAULT NULL,
  `ciggerate_detection` tinyint(1) DEFAULT NULL,
  `sit_detection` tinyint(1) DEFAULT NULL,
  `cam_start` varchar(200) DEFAULT NULL,
  `cam_stop` varchar(200) DEFAULT NULL,
  `attendance_time_start` varchar(200) DEFAULT NULL,
  `attendance_time_end` varchar(200) DEFAULT NULL,
  `leaving_time_start` varchar(200) DEFAULT NULL,
  `leaving_time_end` varchar(200) DEFAULT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  `role_camera` varchar(10) NOT NULL,
  `company_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `camera_settings`
--

INSERT INTO `camera_settings` (`id`, `cam_name`, `feed_src`, `x1`, `y1`, `x2`, `y2`, `x3`, `y3`, `x4`, `y4`, `x5`, `y5`, `x6`, `y6`, `x7`, `y7`, `x8`, `y8`, `cam_is_active`, `gender_detection`, `face_detection`, `face_capture`, `id_card_detection`, `uniform_detection`, `shoes_detection`, `ciggerate_detection`, `sit_detection`, `cam_start`, `cam_stop`, `attendance_time_start`, `attendance_time_end`, `leaving_time_start`, `leaving_time_end`, `createdAt`, `updatedAt`, `role_camera`, `company_id`) VALUES
(2, 'ruangan kerja', 'rtsp://admin:admin@10.2.4.222:8554/Streaming/Channels/102', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-05-30 14:35:15', '2025-07-08 13:05:02', 'T', 1),
(4, 'Ruang Aula', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-06-03 09:57:47', '2025-06-03 09:57:47', 'T', 1),
(5, 'Ruang Santai', 'rtsp://192.168.61.14:8080/h264_ulaw.sdp', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-06-03 15:02:41', '2025-06-03 15:05:04', 'T', 1),
(6, 'Kamera Rufa', '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', '23:11:00', '23:20:00', '23:30:00', '23:55:00', '2025-06-15 23:12:36', '2025-06-15 23:33:58', 'P', 1),
(7, 'Kamera Aula 2', '0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-06-20 14:44:30', '2025-06-20 14:44:30', 'T', 1),
(8, 'kamera dalam', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-06-30 17:26:40', '2025-06-30 17:26:40', 'T', 3),
(9, 'Kamera Luar', 'rtsp://admin:bismillah9x@192.168.4.3:554/Streaming/Channels/101', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-07-06 19:21:15', '2025-07-15 11:12:45', 'T', 1),
(10, 'Kamera AI', 'rtsp://admin:admin@192.168.1.28:8554/Streaming/Channels/102', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, '00:01:00', '23:59:00', NULL, NULL, NULL, NULL, '2025-07-07 22:25:18', '2025-07-07 22:25:18', 'T', 1);

-- --------------------------------------------------------

--
-- Struktur dari tabel `company`
--

CREATE TABLE `company` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `company`
--

INSERT INTO `company` (`id`, `name`, `createdAt`, `updatedAt`, `user_id`) VALUES
(1, 'UNS PSDKU', '2025-05-29 07:10:56', '2025-05-29 07:10:56', 2),
(2, 'garapan', '2025-06-03 15:15:20', '2025-06-03 15:15:20', 4),
(3, 'BISA AI', '2025-06-30 13:09:46', '2025-06-30 13:09:46', 6);

-- --------------------------------------------------------

--
-- Struktur dari tabel `counted_instances`
--

CREATE TABLE `counted_instances` (
  `id` int(11) NOT NULL,
  `timestamp` datetime NOT NULL,
  `camera_id` int(11) NOT NULL,
  `male_entries` int(11) DEFAULT NULL,
  `female_entries` int(11) DEFAULT NULL,
  `unknown_gender_entries` int(11) DEFAULT NULL,
  `staff_entries` int(11) DEFAULT NULL,
  `intern_entries` int(11) DEFAULT NULL,
  `unknown_role_entries` int(11) DEFAULT NULL,
  `people_exits` int(11) DEFAULT NULL,
  `people_inside` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `divisions`
--

CREATE TABLE `divisions` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `company_id` int(11) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `divisions`
--

INSERT INTO `divisions` (`id`, `name`, `company_id`, `createdAt`, `updatedAt`) VALUES
(1, 'IT', 1, '2025-05-30 08:30:25', '2025-05-30 08:30:25'),
(2, 'Cyber Security', 3, '2025-06-30 13:12:29', '2025-06-30 13:12:29'),
(3, 'Marketing', 1, '2025-07-07 22:17:12', '2025-07-07 22:17:12');

-- --------------------------------------------------------

--
-- Struktur dari tabel `personnels`
--

CREATE TABLE `personnels` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  `division_id` int(11) DEFAULT NULL,
  `company_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `personnels`
--

INSERT INTO `personnels` (`id`, `name`, `createdAt`, `updatedAt`, `user_id`, `division_id`, `company_id`) VALUES
(1, 'subekti', '2025-05-30 08:30:54', '2025-05-30 08:30:54', 3, 1, 1),
(2, 'Rafi', '2025-06-16 08:49:31', '2025-06-16 08:49:31', 5, 1, 1),
(3, 'mahesa agung', '2025-06-30 13:13:16', '2025-06-30 15:04:00', 7, 2, 3),
(4, 'fafa', '2025-06-30 15:24:22', '2025-06-30 15:24:22', 8, 1, 1),
(5, 'muhammad abidin', '2025-06-30 17:13:41', '2025-06-30 17:13:41', 9, 1, 1),
(6, 'fabianus', '2025-06-30 18:02:20', '2025-06-30 18:02:20', 10, 1, 1),
(7, 'surya', '2025-07-03 19:36:15', '2025-07-03 19:36:15', 11, 1, 1),
(8, 'Ella Wandasari', '2025-07-07 22:21:23', '2025-07-07 22:21:23', 12, 3, 1);

-- --------------------------------------------------------

--
-- Struktur dari tabel `personnel_entries`
--

CREATE TABLE `personnel_entries` (
  `id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  `timestamp` datetime NOT NULL,
  `presence_status` varchar(10) NOT NULL,
  `personnel_id` int(11) NOT NULL,
  `image` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `personnel_entries`
--

INSERT INTO `personnel_entries` (`id`, `camera_id`, `timestamp`, `presence_status`, `personnel_id`, `image`) VALUES
(1, 6, '2025-06-15 23:25:45', 'ONTIME', 1, 'img/presence_proofs/20250615/1_subekti_232545204529_91.jpg'),
(2, 6, '2025-06-15 23:34:41', 'LEAVE', 1, 'img/presence_proofs/20250615/1_subekti_233441778828_92.jpg');

-- --------------------------------------------------------

--
-- Struktur dari tabel `personnel_images`
--

CREATE TABLE `personnel_images` (
  `id` int(11) NOT NULL,
  `personnel_id` int(11) NOT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `createdAt` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `personnel_images`
--

INSERT INTO `personnel_images` (`id`, `personnel_id`, `image_path`, `createdAt`) VALUES
(1, 1, 'personnel_pics/subekti/face_1_subekti_1.jpg', '2025-05-30 08:31:49'),
(2, 1, 'personnel_pics/subekti/face_1_subekti_2.jpg', '2025-05-30 08:31:49'),
(3, 1, 'personnel_pics/subekti/face_1_subekti_3.jpg', '2025-05-30 08:31:49'),
(4, 1, 'personnel_pics/subekti/face_1_subekti_4.jpg', '2025-05-30 08:31:49'),
(5, 1, 'personnel_pics/subekti/face_1_subekti_5.jpg', '2025-05-30 08:31:49'),
(6, 1, 'personnel_pics/subekti/face_1_subekti_6.jpg', '2025-05-30 08:31:49'),
(7, 1, 'personnel_pics/subekti/face_1_subekti_7.jpg', '2025-05-30 08:31:49'),
(8, 1, 'personnel_pics/subekti/face_1_subekti_8.jpg', '2025-05-30 08:31:49'),
(9, 1, 'personnel_pics/subekti/face_1_subekti_9.jpg', '2025-05-30 08:31:49'),
(10, 1, 'personnel_pics/subekti/face_1_subekti_10.jpg', '2025-05-30 08:31:49'),
(11, 1, 'personnel_pics/subekti/face_1_subekti_11.jpg', '2025-05-30 08:31:49'),
(12, 1, 'personnel_pics/subekti/face_1_subekti_12.jpg', '2025-05-30 08:31:49'),
(13, 1, 'personnel_pics/subekti/face_1_subekti_13.jpg', '2025-05-30 08:31:49'),
(14, 1, 'personnel_pics/subekti/face_1_subekti_14.jpg', '2025-05-30 08:31:49'),
(15, 1, 'personnel_pics/subekti/face_1_subekti_15.jpg', '2025-05-30 08:31:49'),
(16, 1, 'personnel_pics/subekti/face_1_subekti_16.jpg', '2025-05-30 08:31:49'),
(17, 1, 'personnel_pics/subekti/face_1_subekti_17.jpg', '2025-05-30 08:31:49'),
(18, 1, 'personnel_pics/subekti/face_1_subekti_18.jpg', '2025-05-30 08:31:49'),
(19, 1, 'personnel_pics/subekti/face_1_subekti_19.jpg', '2025-05-30 08:31:49'),
(20, 1, 'personnel_pics/subekti/face_1_subekti_20.jpg', '2025-05-30 08:31:49'),
(21, 1, 'personnel_pics/subekti/face_1_subekti_21.jpg', '2025-05-30 08:31:49'),
(22, 1, 'personnel_pics/subekti/face_1_subekti_22.jpg', '2025-05-30 08:31:49'),
(23, 1, 'personnel_pics/subekti/face_1_subekti_23.jpg', '2025-05-30 08:31:49'),
(24, 1, 'personnel_pics/subekti/face_1_subekti_24.jpg', '2025-05-30 08:31:49'),
(25, 1, 'personnel_pics/subekti/face_1_subekti_25.jpg', '2025-05-30 08:31:49'),
(26, 1, 'personnel_pics/subekti/face_1_subekti_26.jpg', '2025-05-30 08:31:49'),
(27, 1, 'personnel_pics/subekti/face_1_subekti_27.jpg', '2025-05-30 08:31:49'),
(28, 1, 'personnel_pics/subekti/face_1_subekti_28.jpg', '2025-05-30 08:31:49'),
(29, 1, 'personnel_pics/subekti/face_1_subekti_29.jpg', '2025-05-30 08:31:49'),
(30, 1, 'personnel_pics/subekti/face_1_subekti_30.jpg', '2025-05-30 08:31:49'),
(31, 1, 'personnel_pics/subekti/face_1_subekti_31.jpg', '2025-05-30 08:31:49'),
(32, 1, 'personnel_pics/subekti/face_1_subekti_32.jpg', '2025-05-30 08:31:49'),
(33, 1, 'personnel_pics/subekti/face_1_subekti_33.jpg', '2025-05-30 08:31:49'),
(34, 1, 'personnel_pics/subekti/face_1_subekti_34.jpg', '2025-05-30 08:31:49'),
(35, 1, 'personnel_pics/subekti/face_1_subekti_35.jpg', '2025-05-30 08:31:49'),
(36, 1, 'personnel_pics/subekti/face_1_subekti_36.jpg', '2025-05-30 08:31:49'),
(37, 1, 'personnel_pics/subekti/face_1_subekti_37.jpg', '2025-05-30 08:31:49'),
(38, 1, 'personnel_pics/subekti/face_1_subekti_38.jpg', '2025-05-30 08:31:49'),
(39, 1, 'personnel_pics/subekti/face_1_subekti_39.jpg', '2025-05-30 08:31:49'),
(40, 1, 'personnel_pics/subekti/face_1_subekti_40.jpg', '2025-05-30 08:31:49'),
(41, 1, 'personnel_pics/subekti/face_1_subekti_41.jpg', '2025-05-30 08:31:49'),
(42, 1, 'personnel_pics/subekti/face_1_subekti_42.jpg', '2025-05-30 08:31:49'),
(43, 1, 'personnel_pics/subekti/face_1_subekti_43.jpg', '2025-05-30 08:31:49'),
(44, 1, 'personnel_pics/subekti/face_1_subekti_44.jpg', '2025-05-30 08:31:49'),
(45, 1, 'personnel_pics/subekti/face_1_subekti_45.jpg', '2025-05-30 08:31:49'),
(46, 1, 'personnel_pics/subekti/face_1_subekti_46.jpg', '2025-05-30 08:31:49'),
(47, 1, 'personnel_pics/subekti/face_1_subekti_47.jpg', '2025-05-30 08:31:49'),
(48, 1, 'personnel_pics/subekti/face_1_subekti_48.jpg', '2025-05-30 08:31:49'),
(49, 1, 'personnel_pics/subekti/face_1_subekti_49.jpg', '2025-05-30 08:31:49'),
(50, 1, 'personnel_pics/subekti/face_1_subekti_50.jpg', '2025-05-30 08:31:49'),
(51, 1, 'personnel_pics/subekti/face_1_subekti_001.jpg', '2025-06-15 23:14:30'),
(52, 1, 'personnel_pics/subekti/face_1_subekti_002.jpg', '2025-06-15 23:14:30'),
(53, 1, 'personnel_pics/subekti/face_1_subekti_003.jpg', '2025-06-15 23:14:30'),
(54, 1, 'personnel_pics/subekti/face_1_subekti_004.jpg', '2025-06-15 23:14:30'),
(56, 1, 'personnel_pics/subekti/face_1_subekti_006.jpg', '2025-06-15 23:14:30'),
(57, 1, 'personnel_pics/subekti/face_1_subekti_007.jpg', '2025-06-15 23:14:30'),
(58, 1, 'personnel_pics/subekti/face_1_subekti_008.jpg', '2025-06-15 23:14:30'),
(59, 1, 'personnel_pics/subekti/face_1_subekti_009.jpg', '2025-06-15 23:14:30'),
(60, 1, 'personnel_pics/subekti/face_1_subekti_010.jpg', '2025-06-15 23:14:30'),
(61, 1, 'personnel_pics/subekti/face_1_subekti_011.jpg', '2025-06-15 23:14:30'),
(62, 1, 'personnel_pics/subekti/face_1_subekti_012.jpg', '2025-06-15 23:14:30'),
(63, 1, 'personnel_pics/subekti/face_1_subekti_013.jpg', '2025-06-15 23:14:30'),
(64, 1, 'personnel_pics/subekti/face_1_subekti_014.jpg', '2025-06-15 23:14:30'),
(65, 1, 'personnel_pics/subekti/face_1_subekti_015.jpg', '2025-06-15 23:14:30'),
(66, 1, 'personnel_pics/subekti/face_1_subekti_016.jpg', '2025-06-15 23:14:30'),
(67, 1, 'personnel_pics/subekti/face_1_subekti_017.jpg', '2025-06-15 23:14:30'),
(68, 1, 'personnel_pics/subekti/face_1_subekti_018.jpg', '2025-06-15 23:14:30'),
(69, 1, 'personnel_pics/subekti/face_1_subekti_019.jpg', '2025-06-15 23:14:30'),
(70, 1, 'personnel_pics/subekti/face_1_subekti_020.jpg', '2025-06-15 23:14:30'),
(71, 1, 'personnel_pics/subekti/face_1_subekti_021.jpg', '2025-06-15 23:14:30'),
(72, 1, 'personnel_pics/subekti/face_1_subekti_022.jpg', '2025-06-15 23:14:30'),
(73, 1, 'personnel_pics/subekti/face_1_subekti_023.jpg', '2025-06-15 23:14:30'),
(74, 1, 'personnel_pics/subekti/face_1_subekti_024.jpg', '2025-06-15 23:14:30'),
(75, 1, 'personnel_pics/subekti/face_1_subekti_025.jpg', '2025-06-15 23:14:30'),
(76, 1, 'personnel_pics/subekti/face_1_subekti_026.jpg', '2025-06-15 23:14:30'),
(77, 1, 'personnel_pics/subekti/face_1_subekti_027.jpg', '2025-06-15 23:14:30'),
(78, 1, 'personnel_pics/subekti/face_1_subekti_028.jpg', '2025-06-15 23:14:30'),
(79, 1, 'personnel_pics/subekti/face_1_subekti_029.jpg', '2025-06-15 23:14:30'),
(80, 1, 'personnel_pics/subekti/face_1_subekti_030.jpg', '2025-06-15 23:14:30'),
(81, 1, 'personnel_pics/subekti/face_1_subekti_031.jpg', '2025-06-15 23:14:30'),
(82, 1, 'personnel_pics/subekti/face_1_subekti_032.jpg', '2025-06-15 23:14:30'),
(83, 1, 'personnel_pics/subekti/face_1_subekti_033.jpg', '2025-06-15 23:14:30'),
(84, 1, 'personnel_pics/subekti/face_1_subekti_034.jpg', '2025-06-15 23:14:30'),
(85, 1, 'personnel_pics/subekti/face_1_subekti_035.jpg', '2025-06-15 23:14:30'),
(86, 1, 'personnel_pics/subekti/face_1_subekti_036.jpg', '2025-06-15 23:14:30'),
(87, 1, 'personnel_pics/subekti/face_1_subekti_037.jpg', '2025-06-15 23:14:30'),
(88, 1, 'personnel_pics/subekti/face_1_subekti_038.jpg', '2025-06-15 23:14:30'),
(89, 1, 'personnel_pics/subekti/face_1_subekti_039.jpg', '2025-06-15 23:14:30'),
(90, 1, 'personnel_pics/subekti/face_1_subekti_040.jpg', '2025-06-15 23:14:30'),
(91, 1, 'personnel_pics/subekti/face_1_subekti_041.jpg', '2025-06-15 23:14:30'),
(92, 1, 'personnel_pics/subekti/face_1_subekti_042.jpg', '2025-06-15 23:14:30'),
(93, 1, 'personnel_pics/subekti/face_1_subekti_043.jpg', '2025-06-15 23:14:30'),
(94, 1, 'personnel_pics/subekti/face_1_subekti_044.jpg', '2025-06-15 23:14:30'),
(95, 1, 'personnel_pics/subekti/face_1_subekti_045.jpg', '2025-06-15 23:14:30'),
(96, 1, 'personnel_pics/subekti/face_1_subekti_046.jpg', '2025-06-15 23:14:30'),
(97, 1, 'personnel_pics/subekti/face_1_subekti_047.jpg', '2025-06-15 23:14:30'),
(98, 1, 'personnel_pics/subekti/face_1_subekti_048.jpg', '2025-06-15 23:14:30'),
(99, 1, 'personnel_pics/subekti/face_1_subekti_049.jpg', '2025-06-15 23:14:30'),
(100, 1, 'personnel_pics/subekti/face_1_subekti_050.jpg', '2025-06-15 23:14:30'),
(101, 2, 'personnel_pics/Rafi/face_2_Rafi_001.jpg', '2025-06-16 08:54:19'),
(102, 2, 'personnel_pics/Rafi/face_2_Rafi_002.jpg', '2025-06-16 08:54:19'),
(103, 2, 'personnel_pics/Rafi/face_2_Rafi_003.jpg', '2025-06-16 08:54:19'),
(104, 2, 'personnel_pics/Rafi/face_2_Rafi_004.jpg', '2025-06-16 08:54:19'),
(105, 2, 'personnel_pics/Rafi/face_2_Rafi_005.jpg', '2025-06-16 08:54:19'),
(106, 2, 'personnel_pics/Rafi/face_2_Rafi_006.jpg', '2025-06-16 08:54:19'),
(107, 2, 'personnel_pics/Rafi/face_2_Rafi_007.jpg', '2025-06-16 08:54:19'),
(108, 2, 'personnel_pics/Rafi/face_2_Rafi_008.jpg', '2025-06-16 08:54:19'),
(109, 2, 'personnel_pics/Rafi/face_2_Rafi_009.jpg', '2025-06-16 08:54:19'),
(110, 2, 'personnel_pics/Rafi/face_2_Rafi_010.jpg', '2025-06-16 08:54:19'),
(111, 2, 'personnel_pics/Rafi/face_2_Rafi_011.jpg', '2025-06-16 08:54:19'),
(112, 2, 'personnel_pics/Rafi/face_2_Rafi_012.jpg', '2025-06-16 08:54:19'),
(113, 2, 'personnel_pics/Rafi/face_2_Rafi_013.jpg', '2025-06-16 08:54:19'),
(114, 2, 'personnel_pics/Rafi/face_2_Rafi_014.jpg', '2025-06-16 08:54:19'),
(115, 2, 'personnel_pics/Rafi/face_2_Rafi_015.jpg', '2025-06-16 08:54:19'),
(116, 2, 'personnel_pics/Rafi/face_2_Rafi_016.jpg', '2025-06-16 08:54:19'),
(117, 2, 'personnel_pics/Rafi/face_2_Rafi_017.jpg', '2025-06-16 08:54:19'),
(118, 2, 'personnel_pics/Rafi/face_2_Rafi_018.jpg', '2025-06-16 08:54:19'),
(119, 2, 'personnel_pics/Rafi/face_2_Rafi_019.jpg', '2025-06-16 08:54:19'),
(120, 2, 'personnel_pics/Rafi/face_2_Rafi_020.jpg', '2025-06-16 08:54:19'),
(121, 2, 'personnel_pics/Rafi/face_2_Rafi_021.jpg', '2025-06-16 08:54:19'),
(122, 2, 'personnel_pics/Rafi/face_2_Rafi_022.jpg', '2025-06-16 08:54:19'),
(123, 2, 'personnel_pics/Rafi/face_2_Rafi_023.jpg', '2025-06-16 08:54:19'),
(124, 2, 'personnel_pics/Rafi/face_2_Rafi_024.jpg', '2025-06-16 08:54:19'),
(125, 2, 'personnel_pics/Rafi/face_2_Rafi_025.jpg', '2025-06-16 08:54:19'),
(126, 2, 'personnel_pics/Rafi/face_2_Rafi_026.jpg', '2025-06-16 08:54:19'),
(127, 2, 'personnel_pics/Rafi/face_2_Rafi_027.jpg', '2025-06-16 08:54:19'),
(128, 2, 'personnel_pics/Rafi/face_2_Rafi_028.jpg', '2025-06-16 08:54:19'),
(129, 2, 'personnel_pics/Rafi/face_2_Rafi_029.jpg', '2025-06-16 08:54:19'),
(130, 2, 'personnel_pics/Rafi/face_2_Rafi_030.jpg', '2025-06-16 08:54:19'),
(131, 2, 'personnel_pics/Rafi/face_2_Rafi_031.jpg', '2025-06-16 08:54:19'),
(132, 2, 'personnel_pics/Rafi/face_2_Rafi_032.jpg', '2025-06-16 08:54:19'),
(133, 2, 'personnel_pics/Rafi/face_2_Rafi_033.jpg', '2025-06-16 08:54:19'),
(134, 2, 'personnel_pics/Rafi/face_2_Rafi_034.jpg', '2025-06-16 08:54:19'),
(135, 2, 'personnel_pics/Rafi/face_2_Rafi_035.jpg', '2025-06-16 08:54:19'),
(136, 2, 'personnel_pics/Rafi/face_2_Rafi_036.jpg', '2025-06-16 08:54:19'),
(137, 2, 'personnel_pics/Rafi/face_2_Rafi_037.jpg', '2025-06-16 08:54:19'),
(138, 2, 'personnel_pics/Rafi/face_2_Rafi_038.jpg', '2025-06-16 08:54:19'),
(139, 2, 'personnel_pics/Rafi/face_2_Rafi_039.jpg', '2025-06-16 08:54:19'),
(140, 2, 'personnel_pics/Rafi/face_2_Rafi_040.jpg', '2025-06-16 08:54:19'),
(141, 2, 'personnel_pics/Rafi/face_2_Rafi_041.jpg', '2025-06-16 08:54:19'),
(142, 2, 'personnel_pics/Rafi/face_2_Rafi_042.jpg', '2025-06-16 08:54:19'),
(143, 2, 'personnel_pics/Rafi/face_2_Rafi_043.jpg', '2025-06-16 08:54:19'),
(144, 2, 'personnel_pics/Rafi/face_2_Rafi_044.jpg', '2025-06-16 08:54:19'),
(145, 2, 'personnel_pics/Rafi/face_2_Rafi_045.jpg', '2025-06-16 08:54:19'),
(146, 2, 'personnel_pics/Rafi/face_2_Rafi_046.jpg', '2025-06-16 08:54:19'),
(147, 2, 'personnel_pics/Rafi/face_2_Rafi_047.jpg', '2025-06-16 08:54:19'),
(148, 2, 'personnel_pics/Rafi/face_2_Rafi_048.jpg', '2025-06-16 08:54:19'),
(149, 2, 'personnel_pics/Rafi/face_2_Rafi_049.jpg', '2025-06-16 08:54:19'),
(150, 2, 'personnel_pics/Rafi/face_2_Rafi_050.jpg', '2025-06-16 08:54:19'),
(201, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_001.jpg', '2025-06-30 15:06:53'),
(202, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_002.jpg', '2025-06-30 15:06:53'),
(203, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_003.jpg', '2025-06-30 15:06:53'),
(204, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_004.jpg', '2025-06-30 15:06:53'),
(205, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_005.jpg', '2025-06-30 15:06:53'),
(206, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_006.jpg', '2025-06-30 15:06:53'),
(207, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_007.jpg', '2025-06-30 15:06:53'),
(208, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_008.jpg', '2025-06-30 15:06:53'),
(209, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_009.jpg', '2025-06-30 15:06:53'),
(210, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_010.jpg', '2025-06-30 15:06:53'),
(211, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_011.jpg', '2025-06-30 15:06:53'),
(212, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_012.jpg', '2025-06-30 15:06:53'),
(213, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_013.jpg', '2025-06-30 15:06:53'),
(214, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_014.jpg', '2025-06-30 15:06:53'),
(215, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_015.jpg', '2025-06-30 15:06:53'),
(216, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_016.jpg', '2025-06-30 15:06:53'),
(217, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_017.jpg', '2025-06-30 15:06:53'),
(218, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_018.jpg', '2025-06-30 15:06:53'),
(219, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_019.jpg', '2025-06-30 15:06:53'),
(220, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_020.jpg', '2025-06-30 15:06:53'),
(221, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_021.jpg', '2025-06-30 15:06:53'),
(222, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_022.jpg', '2025-06-30 15:06:53'),
(223, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_023.jpg', '2025-06-30 15:06:53'),
(224, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_024.jpg', '2025-06-30 15:06:53'),
(225, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_025.jpg', '2025-06-30 15:06:53'),
(226, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_026.jpg', '2025-06-30 15:06:53'),
(227, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_027.jpg', '2025-06-30 15:06:53'),
(228, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_028.jpg', '2025-06-30 15:06:53'),
(229, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_029.jpg', '2025-06-30 15:06:53'),
(230, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_030.jpg', '2025-06-30 15:06:53'),
(231, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_031.jpg', '2025-06-30 15:06:53'),
(232, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_032.jpg', '2025-06-30 15:06:53'),
(233, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_033.jpg', '2025-06-30 15:06:53'),
(234, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_034.jpg', '2025-06-30 15:06:53'),
(235, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_035.jpg', '2025-06-30 15:06:53'),
(236, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_036.jpg', '2025-06-30 15:06:53'),
(237, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_037.jpg', '2025-06-30 15:06:53'),
(238, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_038.jpg', '2025-06-30 15:06:53'),
(239, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_039.jpg', '2025-06-30 15:06:53'),
(240, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_040.jpg', '2025-06-30 15:06:53'),
(241, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_041.jpg', '2025-06-30 15:06:53'),
(242, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_042.jpg', '2025-06-30 15:06:53'),
(243, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_043.jpg', '2025-06-30 15:06:53'),
(244, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_044.jpg', '2025-06-30 15:06:53'),
(245, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_045.jpg', '2025-06-30 15:06:53'),
(246, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_046.jpg', '2025-06-30 15:06:53'),
(247, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_047.jpg', '2025-06-30 15:06:53'),
(248, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_048.jpg', '2025-06-30 15:06:53'),
(249, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_049.jpg', '2025-06-30 15:06:53'),
(250, 3, 'personnel_pics/mahesa agung/face_3_mahesa agung_050.jpg', '2025-06-30 15:06:53'),
(282, 4, 'personnel_pics/fafa/face_4_fafa_001.jpg', '2025-06-30 16:03:57'),
(283, 4, 'personnel_pics/fafa/face_4_fafa_002.jpg', '2025-06-30 16:03:57'),
(284, 4, 'personnel_pics/fafa/face_4_fafa_003.jpg', '2025-06-30 16:03:57'),
(285, 4, 'personnel_pics/fafa/face_4_fafa_004.jpg', '2025-06-30 16:03:57'),
(286, 4, 'personnel_pics/fafa/face_4_fafa_005.jpg', '2025-06-30 16:03:57'),
(287, 4, 'personnel_pics/fafa/face_4_fafa_006.jpg', '2025-06-30 16:03:57'),
(288, 4, 'personnel_pics/fafa/face_4_fafa_007.jpg', '2025-06-30 16:03:57'),
(289, 4, 'personnel_pics/fafa/face_4_fafa_008.jpg', '2025-06-30 16:03:57'),
(290, 4, 'personnel_pics/fafa/face_4_fafa_009.jpg', '2025-06-30 16:03:57'),
(291, 4, 'personnel_pics/fafa/face_4_fafa_010.jpg', '2025-06-30 16:03:57'),
(292, 4, 'personnel_pics/fafa/face_4_fafa_011.jpg', '2025-06-30 16:03:57'),
(293, 4, 'personnel_pics/fafa/face_4_fafa_012.jpg', '2025-06-30 16:03:57'),
(294, 4, 'personnel_pics/fafa/face_4_fafa_013.jpg', '2025-06-30 16:03:57'),
(295, 4, 'personnel_pics/fafa/face_4_fafa_014.jpg', '2025-06-30 16:03:57'),
(296, 4, 'personnel_pics/fafa/face_4_fafa_015.jpg', '2025-06-30 16:03:57'),
(297, 4, 'personnel_pics/fafa/face_4_fafa_016.jpg', '2025-06-30 16:03:57'),
(298, 4, 'personnel_pics/fafa/face_4_fafa_017.jpg', '2025-06-30 16:03:57'),
(299, 4, 'personnel_pics/fafa/face_4_fafa_018.jpg', '2025-06-30 16:03:57'),
(300, 4, 'personnel_pics/fafa/face_4_fafa_019.jpg', '2025-06-30 16:03:57'),
(301, 4, 'personnel_pics/fafa/face_4_fafa_020.jpg', '2025-06-30 16:03:57'),
(304, 4, 'personnel_pics/fafa/face_4_fafa_023.jpg', '2025-06-30 16:04:26'),
(306, 4, 'personnel_pics/fafa/face_4_fafa_025.jpg', '2025-06-30 16:04:26'),
(307, 4, 'personnel_pics/fafa/face_4_fafa_026.jpg', '2025-06-30 16:05:02'),
(308, 4, 'personnel_pics/fafa/face_4_fafa_027.jpg', '2025-06-30 16:05:02'),
(309, 4, 'personnel_pics/fafa/face_4_fafa_028.jpg', '2025-06-30 16:05:02'),
(310, 4, 'personnel_pics/fafa/face_4_fafa_029.jpg', '2025-06-30 16:05:02'),
(311, 4, 'personnel_pics/fafa/face_4_fafa_030.jpg', '2025-06-30 16:05:02'),
(312, 4, 'personnel_pics/fafa/face_4_fafa_031.jpg', '2025-06-30 16:05:02'),
(313, 4, 'personnel_pics/fafa/face_4_fafa_032.jpg', '2025-06-30 16:05:02'),
(314, 4, 'personnel_pics/fafa/face_4_fafa_033.jpg', '2025-06-30 16:05:02'),
(315, 4, 'personnel_pics/fafa/face_4_fafa_034.jpg', '2025-06-30 16:05:02'),
(316, 4, 'personnel_pics/fafa/face_4_fafa_035.jpg', '2025-06-30 16:05:02'),
(317, 4, 'personnel_pics/fafa/face_4_fafa_036.jpg', '2025-06-30 16:05:02'),
(318, 4, 'personnel_pics/fafa/face_4_fafa_037.jpg', '2025-06-30 16:05:02'),
(319, 4, 'personnel_pics/fafa/face_4_fafa_038.jpg', '2025-06-30 16:05:02'),
(320, 4, 'personnel_pics/fafa/face_4_fafa_039.jpg', '2025-06-30 16:05:02'),
(321, 4, 'personnel_pics/fafa/face_4_fafa_040.jpg', '2025-06-30 16:05:02'),
(322, 4, 'personnel_pics/fafa/face_4_fafa_041.jpg', '2025-06-30 16:05:02'),
(323, 4, 'personnel_pics/fafa/face_4_fafa_042.jpg', '2025-06-30 16:05:02'),
(324, 4, 'personnel_pics/fafa/face_4_fafa_043.jpg', '2025-06-30 16:05:02'),
(325, 4, 'personnel_pics/fafa/face_4_fafa_044.jpg', '2025-06-30 16:05:02'),
(326, 4, 'personnel_pics/fafa/face_4_fafa_045.jpg', '2025-06-30 16:05:02'),
(357, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_001.jpg', '2025-06-30 17:15:53'),
(358, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_002.jpg', '2025-06-30 17:15:53'),
(359, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_003.jpg', '2025-06-30 17:15:53'),
(360, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_004.jpg', '2025-06-30 17:15:53'),
(361, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_005.jpg', '2025-06-30 17:15:53'),
(362, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_006.jpg', '2025-06-30 17:15:53'),
(363, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_007.jpg', '2025-06-30 17:15:53'),
(364, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_008.jpg', '2025-06-30 17:15:53'),
(365, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_009.jpg', '2025-06-30 17:15:53'),
(366, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_010.jpg', '2025-06-30 17:15:53'),
(367, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_011.jpg', '2025-06-30 17:15:53'),
(368, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_012.jpg', '2025-06-30 17:15:53'),
(369, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_013.jpg', '2025-06-30 17:15:53'),
(370, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_014.jpg', '2025-06-30 17:15:53'),
(371, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_015.jpg', '2025-06-30 17:15:53'),
(372, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_016.jpg', '2025-06-30 17:15:53'),
(373, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_017.jpg', '2025-06-30 17:15:53'),
(374, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_018.jpg', '2025-06-30 17:15:53'),
(375, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_019.jpg', '2025-06-30 17:15:53'),
(376, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_020.jpg', '2025-06-30 17:15:53'),
(377, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_021.jpg', '2025-06-30 17:18:46'),
(378, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_022.jpg', '2025-06-30 17:18:46'),
(379, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_023.jpg', '2025-06-30 17:18:46'),
(380, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_024.jpg', '2025-06-30 17:18:46'),
(381, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_025.jpg', '2025-06-30 17:18:46'),
(382, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_026.jpg', '2025-06-30 17:18:46'),
(383, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_027.jpg', '2025-06-30 17:18:46'),
(384, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_028.jpg', '2025-06-30 17:18:46'),
(385, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_029.jpg', '2025-06-30 17:18:46'),
(386, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_030.jpg', '2025-06-30 17:18:46'),
(387, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_031.jpg', '2025-06-30 17:18:46'),
(388, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_032.jpg', '2025-06-30 17:18:46'),
(389, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_033.jpg', '2025-06-30 17:18:46'),
(390, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_034.jpg', '2025-06-30 17:18:46'),
(391, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_035.jpg', '2025-06-30 17:18:46'),
(392, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_036.jpg', '2025-06-30 17:18:46'),
(393, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_037.jpg', '2025-06-30 17:18:46'),
(394, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_038.jpg', '2025-06-30 17:18:46'),
(395, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_039.jpg', '2025-06-30 17:18:46'),
(396, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_040.jpg', '2025-06-30 17:18:46'),
(397, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_041.jpg', '2025-06-30 17:19:30'),
(398, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_042.jpg', '2025-06-30 17:19:30'),
(399, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_043.jpg', '2025-06-30 17:19:30'),
(400, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_044.jpg', '2025-06-30 17:19:30'),
(401, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_045.jpg', '2025-06-30 17:19:30'),
(402, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_046.jpg', '2025-06-30 17:19:30'),
(403, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_047.jpg', '2025-06-30 17:19:30'),
(404, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_048.jpg', '2025-06-30 17:19:30'),
(405, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_049.jpg', '2025-06-30 17:19:30'),
(406, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_050.jpg', '2025-06-30 17:19:30'),
(407, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_051.jpg', '2025-06-30 17:19:30'),
(408, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_052.jpg', '2025-06-30 17:19:30'),
(409, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_053.jpg', '2025-06-30 17:19:30'),
(410, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_054.jpg', '2025-06-30 17:19:30'),
(411, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_055.jpg', '2025-06-30 17:19:30'),
(412, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_056.jpg', '2025-06-30 17:19:30'),
(413, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_057.jpg', '2025-06-30 17:19:30'),
(414, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_058.jpg', '2025-06-30 17:19:30'),
(415, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_059.jpg', '2025-06-30 17:19:30'),
(416, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_060.jpg', '2025-06-30 17:19:30'),
(417, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_061.jpg', '2025-06-30 17:20:06'),
(418, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_062.jpg', '2025-06-30 17:20:06'),
(419, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_063.jpg', '2025-06-30 17:20:06'),
(420, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_064.jpg', '2025-06-30 17:20:06'),
(421, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_065.jpg', '2025-06-30 17:20:06'),
(422, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_066.jpg', '2025-06-30 17:20:06'),
(423, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_067.jpg', '2025-06-30 17:20:06'),
(424, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_068.jpg', '2025-06-30 17:20:06'),
(425, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_069.jpg', '2025-06-30 17:20:06'),
(426, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_070.jpg', '2025-06-30 17:20:06'),
(427, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_071.jpg', '2025-06-30 17:20:06'),
(428, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_072.jpg', '2025-06-30 17:20:06'),
(429, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_073.jpg', '2025-06-30 17:20:06'),
(430, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_074.jpg', '2025-06-30 17:20:06'),
(431, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_075.jpg', '2025-06-30 17:20:06'),
(432, 5, 'personnel_pics/muhammad abidin/face_5_muhammad abidin_076.jpg', '2025-06-30 17:20:06'),
(534, 7, 'personnel_pics/surya/face_7_surya_002.jpg', '2025-07-03 19:47:16'),
(535, 7, 'personnel_pics/surya/face_7_surya_003.jpg', '2025-07-03 19:47:16'),
(536, 7, 'personnel_pics/surya/face_7_surya_004.jpg', '2025-07-03 19:47:16'),
(537, 7, 'personnel_pics/surya/face_7_surya_005.jpg', '2025-07-03 19:47:16'),
(538, 7, 'personnel_pics/surya/face_7_surya_006.jpg', '2025-07-03 19:47:16'),
(539, 7, 'personnel_pics/surya/face_7_surya_007.jpg', '2025-07-03 19:47:16'),
(540, 7, 'personnel_pics/surya/face_7_surya_008.jpg', '2025-07-03 19:47:16'),
(541, 7, 'personnel_pics/surya/face_7_surya_009.jpg', '2025-07-03 19:47:16'),
(542, 7, 'personnel_pics/surya/face_7_surya_010.jpg', '2025-07-03 19:47:16'),
(543, 7, 'personnel_pics/surya/face_7_surya_011.jpg', '2025-07-03 19:47:16'),
(544, 7, 'personnel_pics/surya/face_7_surya_012.jpg', '2025-07-03 19:47:16'),
(545, 7, 'personnel_pics/surya/face_7_surya_013.jpg', '2025-07-03 19:47:16'),
(546, 7, 'personnel_pics/surya/face_7_surya_014.jpg', '2025-07-03 19:47:16'),
(547, 7, 'personnel_pics/surya/face_7_surya_015.jpg', '2025-07-03 19:47:16'),
(548, 7, 'personnel_pics/surya/face_7_surya_016.jpg', '2025-07-03 19:47:54'),
(549, 7, 'personnel_pics/surya/face_7_surya_017.jpg', '2025-07-03 19:47:54'),
(550, 7, 'personnel_pics/surya/face_7_surya_018.jpg', '2025-07-03 19:47:54'),
(551, 7, 'personnel_pics/surya/face_7_surya_019.jpg', '2025-07-03 19:47:54'),
(552, 7, 'personnel_pics/surya/face_7_surya_020.jpg', '2025-07-03 19:47:54'),
(553, 7, 'personnel_pics/surya/face_7_surya_021.jpg', '2025-07-03 19:47:54'),
(554, 7, 'personnel_pics/surya/face_7_surya_022.jpg', '2025-07-03 19:47:54'),
(555, 7, 'personnel_pics/surya/face_7_surya_023.jpg', '2025-07-03 19:47:54'),
(556, 7, 'personnel_pics/surya/face_7_surya_024.jpg', '2025-07-03 19:47:54'),
(557, 7, 'personnel_pics/surya/face_7_surya_025.jpg', '2025-07-03 19:47:54'),
(558, 7, 'personnel_pics/surya/face_7_surya_026.jpg', '2025-07-03 19:47:54'),
(559, 7, 'personnel_pics/surya/face_7_surya_027.jpg', '2025-07-03 19:47:54'),
(560, 7, 'personnel_pics/surya/face_7_surya_028.jpg', '2025-07-03 19:47:54'),
(561, 7, 'personnel_pics/surya/face_7_surya_029.jpg', '2025-07-03 19:47:54'),
(562, 7, 'personnel_pics/surya/face_7_surya_030.jpg', '2025-07-03 19:47:54'),
(563, 7, 'personnel_pics/surya/face_7_surya_031.jpg', '2025-07-03 19:47:54'),
(564, 7, 'personnel_pics/surya/face_7_surya_032.jpg', '2025-07-03 19:47:54'),
(565, 7, 'personnel_pics/surya/face_7_surya_033.jpg', '2025-07-03 19:47:54'),
(566, 7, 'personnel_pics/surya/face_7_surya_034.jpg', '2025-07-03 19:47:54'),
(567, 7, 'personnel_pics/surya/face_7_surya_035.jpg', '2025-07-03 19:47:54'),
(568, 7, 'personnel_pics/surya/face_7_surya_036.jpg', '2025-07-03 19:48:35'),
(569, 7, 'personnel_pics/surya/face_7_surya_037.jpg', '2025-07-03 19:48:35'),
(570, 7, 'personnel_pics/surya/face_7_surya_038.jpg', '2025-07-03 19:48:35'),
(571, 7, 'personnel_pics/surya/face_7_surya_039.jpg', '2025-07-03 19:48:35'),
(572, 7, 'personnel_pics/surya/face_7_surya_040.jpg', '2025-07-03 19:48:35'),
(573, 7, 'personnel_pics/surya/face_7_surya_041.jpg', '2025-07-03 19:48:35'),
(574, 7, 'personnel_pics/surya/face_7_surya_042.jpg', '2025-07-03 19:48:35'),
(575, 7, 'personnel_pics/surya/face_7_surya_043.jpg', '2025-07-03 19:48:35'),
(576, 7, 'personnel_pics/surya/face_7_surya_044.jpg', '2025-07-03 19:48:35'),
(577, 7, 'personnel_pics/surya/face_7_surya_045.jpg', '2025-07-03 19:48:35'),
(578, 7, 'personnel_pics/surya/face_7_surya_046.jpg', '2025-07-03 19:48:35'),
(579, 7, 'personnel_pics/surya/face_7_surya_047.jpg', '2025-07-03 19:48:35'),
(580, 7, 'personnel_pics/surya/face_7_surya_048.jpg', '2025-07-03 19:48:35'),
(581, 7, 'personnel_pics/surya/face_7_surya_049.jpg', '2025-07-03 19:48:35'),
(582, 7, 'personnel_pics/surya/face_7_surya_050.jpg', '2025-07-03 19:48:35'),
(583, 7, 'personnel_pics/surya/face_7_surya_051.jpg', '2025-07-03 19:48:35'),
(584, 7, 'personnel_pics/surya/face_7_surya_052.jpg', '2025-07-03 19:48:35'),
(585, 7, 'personnel_pics/surya/face_7_surya_053.jpg', '2025-07-03 19:48:35'),
(586, 7, 'personnel_pics/surya/face_7_surya_054.jpg', '2025-07-03 19:48:35'),
(587, 7, 'personnel_pics/surya/face_7_surya_055.jpg', '2025-07-03 19:48:35'),
(588, 7, 'personnel_pics/surya/face_7_surya_056.jpg', '2025-07-03 19:49:13'),
(589, 7, 'personnel_pics/surya/face_7_surya_057.jpg', '2025-07-03 19:49:13'),
(590, 7, 'personnel_pics/surya/face_7_surya_058.jpg', '2025-07-03 19:49:13'),
(591, 7, 'personnel_pics/surya/face_7_surya_059.jpg', '2025-07-03 19:49:13'),
(592, 7, 'personnel_pics/surya/face_7_surya_060.jpg', '2025-07-03 19:49:13'),
(593, 7, 'personnel_pics/surya/face_7_surya_061.jpg', '2025-07-03 19:49:13'),
(594, 7, 'personnel_pics/surya/face_7_surya_062.jpg', '2025-07-03 19:49:13'),
(595, 7, 'personnel_pics/surya/face_7_surya_063.jpg', '2025-07-03 19:49:13'),
(596, 7, 'personnel_pics/surya/face_7_surya_064.jpg', '2025-07-03 19:49:13'),
(597, 7, 'personnel_pics/surya/face_7_surya_065.jpg', '2025-07-03 19:49:13'),
(598, 7, 'personnel_pics/surya/face_7_surya_066.jpg', '2025-07-03 19:49:13'),
(599, 7, 'personnel_pics/surya/face_7_surya_067.jpg', '2025-07-03 19:49:13'),
(600, 7, 'personnel_pics/surya/face_7_surya_068.jpg', '2025-07-03 19:49:13'),
(601, 7, 'personnel_pics/surya/face_7_surya_069.jpg', '2025-07-03 19:49:13'),
(602, 7, 'personnel_pics/surya/face_7_surya_070.jpg', '2025-07-03 19:49:13'),
(603, 7, 'personnel_pics/surya/face_7_surya_071.jpg', '2025-07-03 19:49:13'),
(604, 7, 'personnel_pics/surya/face_7_surya_072.jpg', '2025-07-03 19:49:13'),
(605, 7, 'personnel_pics/surya/face_7_surya_073.jpg', '2025-07-03 19:49:13'),
(606, 7, 'personnel_pics/surya/face_7_surya_074.jpg', '2025-07-03 19:49:13'),
(607, 7, 'personnel_pics/surya/face_7_surya_075.jpg', '2025-07-03 19:49:53'),
(608, 7, 'personnel_pics/surya/face_7_surya_076.jpg', '2025-07-03 19:49:53'),
(609, 7, 'personnel_pics/surya/face_7_surya_077.jpg', '2025-07-03 19:49:53'),
(610, 7, 'personnel_pics/surya/face_7_surya_078.jpg', '2025-07-03 19:49:53'),
(611, 7, 'personnel_pics/surya/face_7_surya_079.jpg', '2025-07-03 19:49:53'),
(612, 7, 'personnel_pics/surya/face_7_surya_080.jpg', '2025-07-03 19:49:53'),
(613, 7, 'personnel_pics/surya/face_7_surya_081.jpg', '2025-07-03 19:49:53'),
(614, 7, 'personnel_pics/surya/face_7_surya_082.jpg', '2025-07-03 19:49:53'),
(615, 7, 'personnel_pics/surya/face_7_surya_083.jpg', '2025-07-03 19:49:53'),
(616, 7, 'personnel_pics/surya/face_7_surya_084.jpg', '2025-07-03 19:49:53'),
(617, 7, 'personnel_pics/surya/face_7_surya_085.jpg', '2025-07-03 19:49:53'),
(618, 7, 'personnel_pics/surya/face_7_surya_086.jpg', '2025-07-03 19:49:53'),
(619, 7, 'personnel_pics/surya/face_7_surya_087.jpg', '2025-07-03 19:49:53'),
(620, 7, 'personnel_pics/surya/face_7_surya_088.jpg', '2025-07-03 19:49:53'),
(621, 7, 'personnel_pics/surya/face_7_surya_089.jpg', '2025-07-03 19:49:53'),
(622, 7, 'personnel_pics/surya/face_7_surya_090.jpg', '2025-07-03 19:49:53'),
(623, 7, 'personnel_pics/surya/face_7_surya_091.jpg', '2025-07-03 19:49:53'),
(624, 7, 'personnel_pics/surya/face_7_surya_092.jpg', '2025-07-03 19:49:53'),
(625, 7, 'personnel_pics/surya/face_7_surya_093.jpg', '2025-07-03 19:49:53'),
(626, 7, 'personnel_pics/surya/face_7_surya_094.jpg', '2025-07-03 19:49:53'),
(627, 1, 'personnel_pics/subekti/face_1_subekti_050.jpg', '2025-07-08 03:14:32'),
(628, 1, 'personnel_pics/subekti/face_1_subekti_051.jpg', '2025-07-08 03:14:32'),
(629, 1, 'personnel_pics/subekti/face_1_subekti_052.jpg', '2025-07-08 03:14:32'),
(630, 1, 'personnel_pics/subekti/face_1_subekti_053.jpg', '2025-07-08 03:14:32'),
(631, 1, 'personnel_pics/subekti/face_1_subekti_054.jpg', '2025-07-08 03:14:32'),
(632, 1, 'personnel_pics/subekti/face_1_subekti_055.jpg', '2025-07-08 03:14:32'),
(633, 1, 'personnel_pics/subekti/face_1_subekti_056.jpg', '2025-07-08 03:14:32'),
(634, 1, 'personnel_pics/subekti/face_1_subekti_057.jpg', '2025-07-08 03:14:32'),
(635, 1, 'personnel_pics/subekti/face_1_subekti_058.jpg', '2025-07-08 03:14:32'),
(636, 1, 'personnel_pics/subekti/face_1_subekti_059.jpg', '2025-07-08 03:14:32'),
(637, 1, 'personnel_pics/subekti/face_1_subekti_060.jpg', '2025-07-08 03:14:32'),
(638, 1, 'personnel_pics/subekti/face_1_subekti_061.jpg', '2025-07-08 03:14:32'),
(639, 1, 'personnel_pics/subekti/face_1_subekti_062.jpg', '2025-07-08 03:14:32'),
(640, 1, 'personnel_pics/subekti/face_1_subekti_063.jpg', '2025-07-08 03:14:32'),
(641, 1, 'personnel_pics/subekti/face_1_subekti_064.jpg', '2025-07-08 03:14:32'),
(642, 1, 'personnel_pics/subekti/face_1_subekti_065.jpg', '2025-07-08 03:14:32'),
(643, 1, 'personnel_pics/subekti/face_1_subekti_066.jpg', '2025-07-08 03:14:32'),
(644, 1, 'personnel_pics/subekti/face_1_subekti_067.jpg', '2025-07-08 03:14:32'),
(645, 1, 'personnel_pics/subekti/face_1_subekti_068.jpg', '2025-07-08 03:14:32'),
(646, 1, 'personnel_pics/subekti/face_1_subekti_069.jpg', '2025-07-08 03:14:32'),
(647, 1, 'personnel_pics/subekti/face_1_subekti_069.jpg', '2025-07-08 03:15:27'),
(648, 1, 'personnel_pics/subekti/face_1_subekti_070.jpg', '2025-07-08 03:15:27'),
(649, 1, 'personnel_pics/subekti/face_1_subekti_071.jpg', '2025-07-08 03:15:27'),
(650, 1, 'personnel_pics/subekti/face_1_subekti_072.jpg', '2025-07-08 03:15:27'),
(651, 1, 'personnel_pics/subekti/face_1_subekti_073.jpg', '2025-07-08 03:15:27'),
(652, 1, 'personnel_pics/subekti/face_1_subekti_074.jpg', '2025-07-08 03:15:27'),
(653, 1, 'personnel_pics/subekti/face_1_subekti_075.jpg', '2025-07-08 03:15:27'),
(654, 1, 'personnel_pics/subekti/face_1_subekti_076.jpg', '2025-07-08 03:15:27'),
(655, 1, 'personnel_pics/subekti/face_1_subekti_077.jpg', '2025-07-08 03:15:27'),
(656, 1, 'personnel_pics/subekti/face_1_subekti_078.jpg', '2025-07-08 03:15:27'),
(657, 1, 'personnel_pics/subekti/face_1_subekti_079.jpg', '2025-07-08 03:15:27'),
(658, 1, 'personnel_pics/subekti/face_1_subekti_080.jpg', '2025-07-08 03:15:27'),
(659, 1, 'personnel_pics/subekti/face_1_subekti_081.jpg', '2025-07-08 03:15:27'),
(660, 1, 'personnel_pics/subekti/face_1_subekti_082.jpg', '2025-07-08 03:15:27'),
(661, 1, 'personnel_pics/subekti/face_1_subekti_083.jpg', '2025-07-08 03:15:27'),
(662, 1, 'personnel_pics/subekti/face_1_subekti_084.jpg', '2025-07-08 03:15:27'),
(663, 1, 'personnel_pics/subekti/face_1_subekti_085.jpg', '2025-07-08 03:15:27'),
(664, 1, 'personnel_pics/subekti/face_1_subekti_086.jpg', '2025-07-08 03:15:27'),
(665, 1, 'personnel_pics/subekti/face_1_subekti_087.jpg', '2025-07-08 03:15:27'),
(666, 1, 'personnel_pics/subekti/face_1_subekti_088.jpg', '2025-07-08 03:15:27'),
(667, 1, 'personnel_pics/subekti/face_1_subekti_088.jpg', '2025-07-08 03:16:23'),
(668, 1, 'personnel_pics/subekti/face_1_subekti_089.jpg', '2025-07-08 03:16:23'),
(669, 1, 'personnel_pics/subekti/face_1_subekti_090.jpg', '2025-07-08 03:16:23'),
(670, 1, 'personnel_pics/subekti/face_1_subekti_091.jpg', '2025-07-08 03:16:23'),
(671, 1, 'personnel_pics/subekti/face_1_subekti_092.jpg', '2025-07-08 03:16:23'),
(672, 1, 'personnel_pics/subekti/face_1_subekti_093.jpg', '2025-07-08 03:16:23'),
(673, 1, 'personnel_pics/subekti/face_1_subekti_094.jpg', '2025-07-08 03:16:23'),
(674, 1, 'personnel_pics/subekti/face_1_subekti_095.jpg', '2025-07-08 03:16:23'),
(675, 1, 'personnel_pics/subekti/face_1_subekti_096.jpg', '2025-07-08 03:16:23'),
(676, 1, 'personnel_pics/subekti/face_1_subekti_097.jpg', '2025-07-08 03:16:23'),
(677, 1, 'personnel_pics/subekti/face_1_subekti_098.jpg', '2025-07-08 03:16:23'),
(678, 1, 'personnel_pics/subekti/face_1_subekti_099.jpg', '2025-07-08 03:16:23'),
(679, 1, 'personnel_pics/subekti/face_1_subekti_100.jpg', '2025-07-08 03:16:23'),
(680, 1, 'personnel_pics/subekti/face_1_subekti_101.jpg', '2025-07-08 03:16:23'),
(681, 1, 'personnel_pics/subekti/face_1_subekti_102.jpg', '2025-07-08 03:16:23'),
(682, 1, 'personnel_pics/subekti/face_1_subekti_103.jpg', '2025-07-08 03:16:23'),
(683, 1, 'personnel_pics/subekti/face_1_subekti_104.jpg', '2025-07-08 03:16:23'),
(684, 1, 'personnel_pics/subekti/face_1_subekti_105.jpg', '2025-07-08 03:16:23'),
(685, 1, 'personnel_pics/subekti/face_1_subekti_106.jpg', '2025-07-08 03:16:23'),
(686, 1, 'personnel_pics/subekti/face_1_subekti_107.jpg', '2025-07-08 03:16:23'),
(687, 1, 'personnel_pics/subekti/face_1_subekti_107.jpg', '2025-07-08 03:17:16'),
(688, 1, 'personnel_pics/subekti/face_1_subekti_108.jpg', '2025-07-08 03:17:16'),
(689, 1, 'personnel_pics/subekti/face_1_subekti_109.jpg', '2025-07-08 03:17:16'),
(690, 1, 'personnel_pics/subekti/face_1_subekti_110.jpg', '2025-07-08 03:17:16'),
(691, 1, 'personnel_pics/subekti/face_1_subekti_111.jpg', '2025-07-08 03:17:16'),
(692, 1, 'personnel_pics/subekti/face_1_subekti_112.jpg', '2025-07-08 03:17:16'),
(693, 1, 'personnel_pics/subekti/face_1_subekti_113.jpg', '2025-07-08 03:17:16'),
(694, 1, 'personnel_pics/subekti/face_1_subekti_114.jpg', '2025-07-08 03:17:16'),
(695, 1, 'personnel_pics/subekti/face_1_subekti_115.jpg', '2025-07-08 03:17:16'),
(696, 1, 'personnel_pics/subekti/face_1_subekti_116.jpg', '2025-07-08 03:17:16'),
(697, 1, 'personnel_pics/subekti/face_1_subekti_117.jpg', '2025-07-08 03:17:16'),
(698, 1, 'personnel_pics/subekti/face_1_subekti_118.jpg', '2025-07-08 03:17:16'),
(699, 1, 'personnel_pics/subekti/face_1_subekti_119.jpg', '2025-07-08 03:17:16'),
(700, 1, 'personnel_pics/subekti/face_1_subekti_120.jpg', '2025-07-08 03:17:16'),
(701, 1, 'personnel_pics/subekti/face_1_subekti_121.jpg', '2025-07-08 03:17:16'),
(702, 1, 'personnel_pics/subekti/face_1_subekti_122.jpg', '2025-07-08 03:17:16'),
(703, 1, 'personnel_pics/subekti/face_1_subekti_123.jpg', '2025-07-08 03:17:16'),
(704, 1, 'personnel_pics/subekti/face_1_subekti_124.jpg', '2025-07-08 03:17:16'),
(705, 1, 'personnel_pics/subekti/face_1_subekti_125.jpg', '2025-07-08 03:17:16'),
(706, 1, 'personnel_pics/subekti/face_1_subekti_126.jpg', '2025-07-08 03:17:16'),
(707, 1, 'personnel_pics/subekti/face_1_subekti_126.jpg', '2025-07-08 03:18:06'),
(708, 1, 'personnel_pics/subekti/face_1_subekti_127.jpg', '2025-07-08 03:18:06'),
(709, 1, 'personnel_pics/subekti/face_1_subekti_128.jpg', '2025-07-08 03:18:06'),
(710, 1, 'personnel_pics/subekti/face_1_subekti_129.jpg', '2025-07-08 03:18:06'),
(711, 1, 'personnel_pics/subekti/face_1_subekti_130.jpg', '2025-07-08 03:18:06'),
(712, 1, 'personnel_pics/subekti/face_1_subekti_131.jpg', '2025-07-08 03:18:06'),
(713, 1, 'personnel_pics/subekti/face_1_subekti_132.jpg', '2025-07-08 03:18:06'),
(714, 1, 'personnel_pics/subekti/face_1_subekti_133.jpg', '2025-07-08 03:18:06'),
(715, 1, 'personnel_pics/subekti/face_1_subekti_134.jpg', '2025-07-08 03:18:06'),
(716, 1, 'personnel_pics/subekti/face_1_subekti_135.jpg', '2025-07-08 03:18:06'),
(717, 1, 'personnel_pics/subekti/face_1_subekti_136.jpg', '2025-07-08 03:18:06'),
(718, 1, 'personnel_pics/subekti/face_1_subekti_137.jpg', '2025-07-08 03:18:06'),
(719, 1, 'personnel_pics/subekti/face_1_subekti_138.jpg', '2025-07-08 03:18:06'),
(720, 1, 'personnel_pics/subekti/face_1_subekti_139.jpg', '2025-07-08 03:18:06'),
(721, 1, 'personnel_pics/subekti/face_1_subekti_140.jpg', '2025-07-08 03:18:06'),
(722, 1, 'personnel_pics/subekti/face_1_subekti_141.jpg', '2025-07-08 03:18:06'),
(723, 1, 'personnel_pics/subekti/face_1_subekti_142.jpg', '2025-07-08 03:18:06'),
(724, 1, 'personnel_pics/subekti/face_1_subekti_143.jpg', '2025-07-08 03:18:06'),
(725, 1, 'personnel_pics/subekti/face_1_subekti_144.jpg', '2025-07-08 03:18:06'),
(726, 1, 'personnel_pics/subekti/face_1_subekti_145.jpg', '2025-07-08 03:18:06'),
(727, 1, 'personnel_pics/subekti/face_1_subekti_001_20250711161910179823.jpg', '2025-07-11 16:19:26'),
(728, 1, 'personnel_pics/subekti/face_1_subekti_002_20250711161911044472.jpg', '2025-07-11 16:19:26'),
(729, 1, 'personnel_pics/subekti/face_1_subekti_003_20250711161911889266.jpg', '2025-07-11 16:19:26'),
(730, 1, 'personnel_pics/subekti/face_1_subekti_004_20250711161912687954.jpg', '2025-07-11 16:19:26'),
(731, 1, 'personnel_pics/subekti/face_1_subekti_005_20250711161913480948.jpg', '2025-07-11 16:19:26'),
(732, 1, 'personnel_pics/subekti/face_1_subekti_006_20250711161914319114.jpg', '2025-07-11 16:19:26'),
(733, 1, 'personnel_pics/subekti/face_1_subekti_007_20250711161915157839.jpg', '2025-07-11 16:19:26'),
(734, 1, 'personnel_pics/subekti/face_1_subekti_008_20250711161916035277.jpg', '2025-07-11 16:19:26'),
(735, 1, 'personnel_pics/subekti/face_1_subekti_009_20250711161916847970.jpg', '2025-07-11 16:19:26'),
(736, 1, 'personnel_pics/subekti/face_1_subekti_010_20250711161917650179.jpg', '2025-07-11 16:19:26'),
(737, 1, 'personnel_pics/subekti/face_1_subekti_011_20250711161918475567.jpg', '2025-07-11 16:19:26'),
(738, 1, 'personnel_pics/subekti/face_1_subekti_012_20250711161919294545.jpg', '2025-07-11 16:19:26'),
(739, 1, 'personnel_pics/subekti/face_1_subekti_013_20250711161920139750.jpg', '2025-07-11 16:19:26'),
(740, 1, 'personnel_pics/subekti/face_1_subekti_014_20250711161920947774.jpg', '2025-07-11 16:19:26'),
(741, 1, 'personnel_pics/subekti/face_1_subekti_015_20250711161921768367.jpg', '2025-07-11 16:19:26'),
(742, 1, 'personnel_pics/subekti/face_1_subekti_016_20250711161922588395.jpg', '2025-07-11 16:19:26'),
(743, 1, 'personnel_pics/subekti/face_1_subekti_017_20250711161923405849.jpg', '2025-07-11 16:19:26'),
(744, 1, 'personnel_pics/subekti/face_1_subekti_018_20250711161924248153.jpg', '2025-07-11 16:19:26'),
(745, 1, 'personnel_pics/subekti/face_1_subekti_019_20250711161925059478.jpg', '2025-07-11 16:19:26'),
(746, 1, 'personnel_pics/subekti/face_1_subekti_020_20250711161925905889.jpg', '2025-07-11 16:19:26'),
(747, 1, 'personnel_pics/subekti/face_1_subekti_021_20250711162244488237.jpg', '2025-07-11 16:23:01'),
(748, 1, 'personnel_pics/subekti/face_1_subekti_022_20250711162245340772.jpg', '2025-07-11 16:23:01'),
(749, 1, 'personnel_pics/subekti/face_1_subekti_023_20250711162246213723.jpg', '2025-07-11 16:23:01'),
(750, 1, 'personnel_pics/subekti/face_1_subekti_024_20250711162247069249.jpg', '2025-07-11 16:23:01'),
(751, 1, 'personnel_pics/subekti/face_1_subekti_025_20250711162247916216.jpg', '2025-07-11 16:23:01'),
(752, 1, 'personnel_pics/subekti/face_1_subekti_026_20250711162248806033.jpg', '2025-07-11 16:23:01'),
(753, 1, 'personnel_pics/subekti/face_1_subekti_027_20250711162249669257.jpg', '2025-07-11 16:23:01'),
(754, 1, 'personnel_pics/subekti/face_1_subekti_028_20250711162250534967.jpg', '2025-07-11 16:23:01'),
(755, 1, 'personnel_pics/subekti/face_1_subekti_029_20250711162251427706.jpg', '2025-07-11 16:23:01'),
(756, 1, 'personnel_pics/subekti/face_1_subekti_030_20250711162252282799.jpg', '2025-07-11 16:23:01'),
(757, 1, 'personnel_pics/subekti/face_1_subekti_031_20250711162253160049.jpg', '2025-07-11 16:23:01'),
(758, 1, 'personnel_pics/subekti/face_1_subekti_032_20250711162254033410.jpg', '2025-07-11 16:23:01'),
(759, 1, 'personnel_pics/subekti/face_1_subekti_033_20250711162254895256.jpg', '2025-07-11 16:23:01'),
(760, 1, 'personnel_pics/subekti/face_1_subekti_034_20250711162255781838.jpg', '2025-07-11 16:23:01'),
(761, 1, 'personnel_pics/subekti/face_1_subekti_035_20250711162256629798.jpg', '2025-07-11 16:23:01'),
(762, 1, 'personnel_pics/subekti/face_1_subekti_036_20250711162257506679.jpg', '2025-07-11 16:23:01'),
(763, 1, 'personnel_pics/subekti/face_1_subekti_037_20250711162258383641.jpg', '2025-07-11 16:23:01'),
(764, 1, 'personnel_pics/subekti/face_1_subekti_038_20250711162259283880.jpg', '2025-07-11 16:23:01'),
(765, 1, 'personnel_pics/subekti/face_1_subekti_039_20250711162300195874.jpg', '2025-07-11 16:23:01'),
(766, 1, 'personnel_pics/subekti/face_1_subekti_040_20250711162301078846.jpg', '2025-07-11 16:23:01');

-- --------------------------------------------------------

--
-- Struktur dari tabel `tracking`
--

CREATE TABLE `tracking` (
  `id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  `detected_class` varchar(100) NOT NULL,
  `confidence` float DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `personnel_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `tracking`
--

INSERT INTO `tracking` (`id`, `camera_id`, `detected_class`, `confidence`, `timestamp`, `image_path`, `personnel_id`) VALUES
(159, 4, 'Tidak memakai dasi', 0.817335, '2025-06-27 15:00:16', 'static/detection_tracking/910bf184030b4db4acea8d3ae596593b.jpg', 1),
(168, 4, 'Tidak Memakai Dasi', 0.68952, '2025-06-30 12:34:03', 'static/detection_tracking/61d9988446c042c2a1ed6213ba1d0a5f.jpg', 1),
(170, 4, 'Tidak Memakai Dasi', 0.616809, '2025-06-30 17:09:05', 'static/detection_tracking/3f3c9b76f30d43aa9e1cdf651a87017f.jpg', 4),
(171, 8, 'Tidak Memakai Dasi', 0.600121, '2025-06-30 17:28:36', 'static/detection_tracking/faf507809f6646e4be5b86cf58b882e5.jpg', 3),
(172, 4, 'Tidak Memakai Dasi', 0.616117, '2025-06-30 17:38:52', 'static/detection_tracking/6350b070a75b45cd80fa58ac75beb03c.jpg', 5),
(173, 4, 'Tidak Memakai Dasi', 0.607696, '2025-07-03 19:31:26', 'static/detection_tracking/6a94152eff394f26b10e813d790ec238.jpg', 1),
(174, 4, 'Tidak Memakai Dasi', 0.672964, '2025-07-03 19:32:48', 'static/detection_tracking/26550fd01dac4b5db787dc54d76a86f4.jpg', 4),
(176, 4, 'Tidak Memakai Dasi', 0.622158, '2025-07-03 20:14:31', 'static/detection_tracking/8d5c0a26067e40d1b00adc20719ace35.jpg', 7),
(207, 7, 'Tidak Memakai Dasi', 0.869955, '2025-07-04 14:49:23', 'static/detection_tracking/e0b80732cd094ecc831f7118b6954a9a.jpg', 1),
(209, 2, 'Tidak Memakai Dasi', 0.743565, '2025-07-07 13:46:51', 'static/detection_tracking/244659bd331e429d8142f54b786c9297.jpg', 1),
(218, 2, 'Tidak Memakai Dasi', 0.707898, '2025-07-07 16:58:17', 'static/detection_tracking/97811a9deec84942aa551c64bff559da.jpg', 3),
(220, 7, 'Tidak Memakai Dasi', 0.838882, '2025-07-08 03:35:32', 'static/detection_tracking/51ed5dbdb8294334b751139399b568e0.jpg', 1),
(221, 7, 'Tidak Memakai Dasi', 0.849407, '2025-07-08 03:35:42', 'static/detection_tracking/21e30b628ead4d6b902de49eeb6b2152.jpg', 4),
(225, 2, 'Tidak Memakai Dasi', 0.770431, '2025-07-08 15:08:36', 'static/detection_tracking/6c6c46154bc8438cabeae74ce97e9707.jpg', 3),
(226, 9, 'Tidak Memakai Dasi', 0.112833, '2025-07-15 10:26:57', 'static/detection_tracking/1e5cb334af344b7f8e00d818405fe6c0.jpg', 1),
(227, 9, 'Tidak Memakai Dasi', 0.112833, '2025-07-15 10:26:57', 'static/detection_tracking/e43b97af62b741e287d3ee95d4c73819.jpg', 2),
(228, 9, 'Tidak Memakai Dasi', 0.112833, '2025-07-15 10:26:57', 'static/detection_tracking/c21ebba8e8cc45ec8ba262580482cf51.jpg', 7);

-- --------------------------------------------------------

--
-- Struktur dari tabel `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `role` varchar(20) NOT NULL,
  `email` varchar(120) DEFAULT NULL,
  `username` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `users`
--

INSERT INTO `users` (`id`, `role`, `email`, `username`, `password_hash`, `createdAt`, `updatedAt`) VALUES
(1, 'superadmin', 'superadmin@example.com', 'superadmin', 'scrypt:32768:8:1$CxKdS3D6IZiCbwyp$6d5066c5300dd4014cd0d4572ac91fc2e9076310c55ab12cc7c64d20dce68e3529e98113ca7a35edc873d0f4233ddd2014187b08b600c2639603ef0becd1dd13', '2025-05-29 07:09:28', '2025-07-07 22:13:13'),
(2, 'admin', 'unspsdku@gmail.com', 'unspsdku', 'scrypt:32768:8:1$qHsoFag3zRDt5eZm$02e18a9c56a1fc84f8bf1036ef887848cf5252fcfc2ab8a77d0f74c16f9bfe3c91de6190c9294b880a379e5f26bab003640c2295be7f5a16c11e6de08aacb262', '2025-05-29 07:10:56', '2025-05-29 07:10:56'),
(3, 'employee', 'subekti@gmail.com', 'subekti', 'scrypt:32768:8:1$Oa2FE8GRWUncYDJX$b847bce76b5ba0a0fb400763e72b971522480578737e7836b91f27ffbd8354532f3c0efc775185310c7e11656a9f98bb67062c197542602f209dae7a1969ff08', '2025-05-30 08:30:54', '2025-05-30 08:30:54'),
(4, 'admin', 'garapan@gmail.com', 'garapan', 'scrypt:32768:8:1$3oLcOfUfccRRFwoM$177830437244558953c83c52ecb585407e44ddfdede5e008d90bc2bececb26b3f79ac4f7776646379351c6396b9f63cd6b2054d0772bb694b125b8a0efb5480f', '2025-06-03 15:15:20', '2025-06-03 15:15:20'),
(5, 'employee', 'rafi@gmail.com', 'rafi', 'scrypt:32768:8:1$ZeQ78j4kR5gVhZp8$91c8322fe0e2b172ddc5ab7869965e469d8a5bcba4b98a9f4118dad6f72520362bc3a9e82e510ae980ef0fa89ca1c9a1b231ba4f49640ddba89a1a08851bad67', '2025-06-16 08:49:31', '2025-06-16 08:49:31'),
(6, 'admin', 'bisaai70@gmail.com', 'bisaaiacademy', 'scrypt:32768:8:1$agiqyasTJUap1eY6$f91afa0f9ec75eb8380b5c41e631bfaab27126e38229bcaa614f2b553bf14fb1d6e33cd7870e949b3d02f339910b13981bd5cb029b65dc4ce77423d89204c7bb', '2025-06-30 13:09:46', '2025-06-30 13:09:46'),
(7, 'employee', 'mahesaagung@student.uns.ac.id', 'mahesa', 'scrypt:32768:8:1$qTVVPTZhwpArCMq4$1bd95227e340f7c25fb963949af1367be5eb712726f5672fd1623111d970ed6865792826a2db18af1f1371cdfe41b4b51e2d61b0504b655090a2b847054e1377', '2025-06-30 13:13:16', '2025-06-30 15:04:00'),
(8, 'employee', 'fafa@gmail.com', 'fafa', 'scrypt:32768:8:1$3UDrU2II0iyfCbvz$5b5827cbff8b914d5093855e8a1c45b4692ab3dfbdf5e2de2c0b7f2294cf0788975439a227b27ceae7fbfc03dcab323491178076fd6c074c2f479282a28b4221', '2025-06-30 15:24:22', '2025-06-30 15:24:22'),
(9, 'employee', 'abidin@gmail.com', 'abidin', 'scrypt:32768:8:1$5dGZupuMxM3cYWk1$4b878b987214bae107670a0a9d3e0c35c3e1503cc8f1750337e8720581c899dbba03bdb2d0c762e32e63039c120a28e75fe5e1741a90b9613821d178a7bbdcb1', '2025-06-30 17:13:41', '2025-06-30 17:13:41'),
(10, 'employee', 'fabianus@gmail.com', 'fabianus', 'scrypt:32768:8:1$2VLcG1UCG3qrMvo8$564548b870486a82ec73e96811c750408ea325d3ff1e658fa197954dabc0f2c151a0de07e899e4d236870bc80ab0710c271497675d222379d9336f8d7c28994a', '2025-06-30 18:02:20', '2025-06-30 18:02:20'),
(11, 'employee', 'surya@gmail.com', 'surya', 'scrypt:32768:8:1$XVc9uSpGMH8hbVwE$2ad93a9a06ebfb2277eabbc0aad52ab0f87a1cf7cfe9e0d919d0227cc3eb7b87bf5a77cd17cbc4caa9a8efa49e0df83e5ac1dbccf63a1c3437528c1bd525dde8', '2025-07-03 19:36:15', '2025-07-03 19:36:15'),
(12, 'employee', 'ella@gmail.com', 'ellawandasari', 'scrypt:32768:8:1$k3OLSHLXWxRHU0hp$e24d04cdeb547b8cd2aab93fe48c7389d1de4be8f66e4712f140515be4e4ceff32729da5c88a04daa62e9887efd787dc9fe87f8d348a3c0af58ffb576fee3222', '2025-07-07 22:21:23', '2025-07-07 22:21:23');

-- --------------------------------------------------------

--
-- Struktur dari tabel `work_timer`
--

CREATE TABLE `work_timer` (
  `id` int(11) NOT NULL,
  `camera_id` int(11) DEFAULT NULL,
  `datetime` datetime NOT NULL,
  `type` varchar(15) NOT NULL,
  `timer` int(11) NOT NULL,
  `personnel_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `alembic_version`
--
ALTER TABLE `alembic_version`
  ADD PRIMARY KEY (`version_num`);

--
-- Indeks untuk tabel `camera_settings`
--
ALTER TABLE `camera_settings`
  ADD PRIMARY KEY (`id`),
  ADD KEY `company_id` (`company_id`);

--
-- Indeks untuk tabel `company`
--
ALTER TABLE `company`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`);

--
-- Indeks untuk tabel `counted_instances`
--
ALTER TABLE `counted_instances`
  ADD PRIMARY KEY (`id`),
  ADD KEY `camera_id` (`camera_id`);

--
-- Indeks untuk tabel `divisions`
--
ALTER TABLE `divisions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `company_id` (`company_id`);

--
-- Indeks untuk tabel `personnels`
--
ALTER TABLE `personnels`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`),
  ADD KEY `company_id` (`company_id`),
  ADD KEY `division_id` (`division_id`);

--
-- Indeks untuk tabel `personnel_entries`
--
ALTER TABLE `personnel_entries`
  ADD PRIMARY KEY (`id`),
  ADD KEY `camera_id` (`camera_id`),
  ADD KEY `personnel_id` (`personnel_id`);

--
-- Indeks untuk tabel `personnel_images`
--
ALTER TABLE `personnel_images`
  ADD PRIMARY KEY (`id`),
  ADD KEY `personnel_id` (`personnel_id`);

--
-- Indeks untuk tabel `tracking`
--
ALTER TABLE `tracking`
  ADD PRIMARY KEY (`id`),
  ADD KEY `camera_id` (`camera_id`),
  ADD KEY `personnel_id` (`personnel_id`);

--
-- Indeks untuk tabel `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indeks untuk tabel `work_timer`
--
ALTER TABLE `work_timer`
  ADD PRIMARY KEY (`id`),
  ADD KEY `camera_id` (`camera_id`),
  ADD KEY `personnel_id` (`personnel_id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `camera_settings`
--
ALTER TABLE `camera_settings`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT untuk tabel `company`
--
ALTER TABLE `company`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT untuk tabel `counted_instances`
--
ALTER TABLE `counted_instances`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT untuk tabel `divisions`
--
ALTER TABLE `divisions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT untuk tabel `personnels`
--
ALTER TABLE `personnels`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT untuk tabel `personnel_entries`
--
ALTER TABLE `personnel_entries`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT untuk tabel `personnel_images`
--
ALTER TABLE `personnel_images`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=767;

--
-- AUTO_INCREMENT untuk tabel `tracking`
--
ALTER TABLE `tracking`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=229;

--
-- AUTO_INCREMENT untuk tabel `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT untuk tabel `work_timer`
--
ALTER TABLE `work_timer`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `camera_settings`
--
ALTER TABLE `camera_settings`
  ADD CONSTRAINT `camera_settings_ibfk_1` FOREIGN KEY (`company_id`) REFERENCES `company` (`id`) ON DELETE CASCADE;

--
-- Ketidakleluasaan untuk tabel `company`
--
ALTER TABLE `company`
  ADD CONSTRAINT `company_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL;

--
-- Ketidakleluasaan untuk tabel `counted_instances`
--
ALTER TABLE `counted_instances`
  ADD CONSTRAINT `counted_instances_ibfk_1` FOREIGN KEY (`camera_id`) REFERENCES `camera_settings` (`id`) ON DELETE CASCADE;

--
-- Ketidakleluasaan untuk tabel `divisions`
--
ALTER TABLE `divisions`
  ADD CONSTRAINT `divisions_ibfk_1` FOREIGN KEY (`company_id`) REFERENCES `company` (`id`) ON DELETE CASCADE;

--
-- Ketidakleluasaan untuk tabel `personnels`
--
ALTER TABLE `personnels`
  ADD CONSTRAINT `personnels_ibfk_1` FOREIGN KEY (`company_id`) REFERENCES `company` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `personnels_ibfk_2` FOREIGN KEY (`division_id`) REFERENCES `divisions` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `personnels_ibfk_3` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL;

--
-- Ketidakleluasaan untuk tabel `personnel_entries`
--
ALTER TABLE `personnel_entries`
  ADD CONSTRAINT `personnel_entries_ibfk_1` FOREIGN KEY (`camera_id`) REFERENCES `camera_settings` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `personnel_entries_ibfk_2` FOREIGN KEY (`personnel_id`) REFERENCES `personnels` (`id`) ON DELETE CASCADE;

--
-- Ketidakleluasaan untuk tabel `personnel_images`
--
ALTER TABLE `personnel_images`
  ADD CONSTRAINT `personnel_images_ibfk_1` FOREIGN KEY (`personnel_id`) REFERENCES `personnels` (`id`) ON DELETE CASCADE;

--
-- Ketidakleluasaan untuk tabel `tracking`
--
ALTER TABLE `tracking`
  ADD CONSTRAINT `tracking_ibfk_1` FOREIGN KEY (`camera_id`) REFERENCES `camera_settings` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `tracking_ibfk_2` FOREIGN KEY (`personnel_id`) REFERENCES `personnels` (`id`) ON DELETE SET NULL;

--
-- Ketidakleluasaan untuk tabel `work_timer`
--
ALTER TABLE `work_timer`
  ADD CONSTRAINT `work_timer_ibfk_1` FOREIGN KEY (`camera_id`) REFERENCES `camera_settings` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `work_timer_ibfk_2` FOREIGN KEY (`personnel_id`) REFERENCES `personnels` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
