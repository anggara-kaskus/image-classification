<?php

if (!($sock = socket_create(AF_INET, SOCK_STREAM, 0))) {
	$errorcode = socket_last_error();
	$errormsg = socket_strerror($errorcode);

	exit("Couldn't create socket: [{$errorcode}] {$errormsg} \n");
}

if (!socket_connect($sock, '127.0.0.1', 1235)) {
	$errorcode = socket_last_error();
	$errormsg = socket_strerror($errorcode);

	exit("Could not connect: [{$errorcode}] {$errormsg} \n");
}

$message = 'https://dl.kaskus.id/cdn-2.tstatic.net/jabar/foto/bank/images/stok-minyak-goreng-kemasan-tiba-tiba-melimpah.jpg';

if (!socket_send($sock, $message, strlen($message), 0)) {
	$errorcode = socket_last_error();
	$errormsg = socket_strerror($errorcode);

	exit("Could not send data: [{$errorcode}] {$errormsg} \n");
}

if (false === socket_recv($sock, $buf, 2045, MSG_WAITALL)) {
	$errorcode = socket_last_error();
	$errormsg = socket_strerror($errorcode);

	exit("Could not receive data: [{$errorcode}] {$errormsg} \n");
}

echo json_encode(json_decode($buf, true), JSON_PRETTY_PRINT);

socket_close($sock);
