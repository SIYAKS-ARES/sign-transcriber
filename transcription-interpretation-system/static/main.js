// TİD Anlamlandırma - Main JavaScript

// Global variables
let isInitialized = false;

// Document ready
$(document).ready(function() {
    initializeApp();
    setupGlobalEventListeners();
});

function initializeApp() {
    console.log('🚀 TİD Anlamlandırma Sistemi başlatılıyor...');
    
    // Toast notifications setup
    setupToastNotifications();
    
    // Global error handling
    setupErrorHandling();
    
    // API base URL
    window.API_BASE = '/api';
    
    isInitialized = true;
    console.log('✅ TİD Anlamlandırma başlatıldı');
}

function setupGlobalEventListeners() {
    // Navigation active state
    updateNavigationState();
    
    // Auto-hide alerts
    setTimeout(function() {
        $('.alert').fadeOut('slow');
    }, 5000);
    
    // Tooltip initialization
    $('[data-bs-toggle="tooltip"]').tooltip();
    
    // Popover initialization
    $('[data-bs-toggle="popover"]').popover();
    
    // Modal form submit handling
    $('#translateModal form').on('submit', function() {
        showToast('Bilgi', 'Çeviri işlemi başlatılıyor...', 'info', 3000);
        $('#translateModal').modal('hide');
    });
}

function setupToastNotifications() {
    // Toast container oluştur
    if ($('#toast-container').length === 0) {
        $('body').append(`
            <div id="toast-container" class="toast-container position-fixed bottom-0 end-0 p-3" style="z-index: 1100;">
            </div>
        `);
    }
}

function setupErrorHandling() {
    // Global AJAX error handler
    $(document).ajaxError(function(event, xhr, settings, thrownError) {
        console.error('AJAX Error:', {
            url: settings.url,
            status: xhr.status,
            error: thrownError,
            response: xhr.responseText
        });
        
        let errorMessage = 'Bilinmeyen hata oluştu';
        
        if (xhr.responseJSON && xhr.responseJSON.error) {
            errorMessage = xhr.responseJSON.error;
        } else if (xhr.status === 0) {
            errorMessage = 'Sunucu bağlantısı kurulamadı';
        } else if (xhr.status === 404) {
            errorMessage = 'API endpoint bulunamadı';
        } else if (xhr.status >= 500) {
            errorMessage = 'Sunucu hatası oluştu';
        }
        
        showToast('Hata', errorMessage, 'error');
    });
    
    // Global JavaScript error handler
    window.onerror = function(msg, url, lineNo, columnNo, error) {
        console.error('JavaScript Hatası:', {
            message: msg,
            source: url,
            line: lineNo,
            column: columnNo,
            error: error
        });
        
        showToast('JavaScript Hatası', 'Bir JavaScript hatası oluştu', 'error');
        return false;
    };
}

function updateNavigationState() {
    const currentPath = window.location.pathname;
    
    $('.navbar-nav .nav-link').removeClass('active');
    
    if (currentPath === '/') {
        $('.navbar-nav .nav-link[href="/"]').addClass('active');
    }
}

// Toast notification system
function showToast(title, message, type = 'info', duration = 5000) {
    const toastId = 'toast-' + Date.now();
    const iconMap = {
        'success': 'fas fa-check-circle text-success',
        'error': 'fas fa-exclamation-circle text-danger',
        'warning': 'fas fa-exclamation-triangle text-warning',
        'info': 'fas fa-info-circle text-info'
    };
    
    const icon = iconMap[type] || iconMap['info'];
    
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <i class="${icon} me-2"></i>
                <strong class="me-auto">${title}</strong>
                <small>${new Date().toLocaleTimeString()}</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    $('#toast-container').append(toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        delay: duration
    });
    
    toast.show();
    
    // Remove from DOM after hide
    toastElement.addEventListener('hidden.bs.toast', function() {
        $(this).remove();
    });
}

// Utility functions
const Utils = {
    // Format confidence as percentage
    formatConfidence: function(confidence) {
        return Math.round(confidence * 100) + '%';
    },
    
    // Format timestamp
    formatTime: function(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    },
    
    // Get confidence color class
    getConfidenceColorClass: function(confidence) {
        if (confidence >= 8) return 'text-success';
        if (confidence >= 6) return 'text-warning';
        return 'text-danger';
    },
    
    // Get confidence badge class
    getConfidenceBadgeClass: function(confidence) {
        if (confidence >= 8) return 'bg-success';
        if (confidence >= 6) return 'bg-warning';
        return 'bg-danger';
    },
    
    // Debounce function
    debounce: function(func, wait, immediate) {
        let timeout;
        return function executedFunction() {
            const context = this;
            const args = arguments;
            const later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    },
    
    // Copy to clipboard
    copyToClipboard: function(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(function() {
                showToast('Başarılı', 'Panoya kopyalandı', 'success', 2000);
            });
        } else {
            // Fallback
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            showToast('Başarılı', 'Panoya kopyalandı', 'success', 2000);
        }
    }
};

// Storage helper functions
const Storage = {
    // Local storage with JSON support
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('LocalStorage write error:', e);
            return false;
        }
    },
    
    get: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('LocalStorage read error:', e);
            return defaultValue;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('LocalStorage remove error:', e);
            return false;
        }
    }
};

// Export for global access
window.TID = {
    Utils,
    Storage,
    showToast
};

// Animation helpers
function animateCountUp(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        $(element).text(Math.floor(current));
    }, 16);
}

// Smooth scroll to element
function scrollToElement(selector, offset = 0) {
    const element = $(selector);
    if (element.length) {
        $('html, body').animate({
            scrollTop: element.offset().top - offset
        }, 800);
    }
}

// Feature card interactions
$(document).on('mouseenter', '.feature-card', function() {
    $(this).find('.feature-icon').addClass('scale-110');
});

$(document).on('mouseleave', '.feature-card', function() {
    $(this).find('.feature-icon').removeClass('scale-110');
});

// Process step interactions
$(document).on('mouseenter', '.process-step', function() {
    $(this).find('.step-number').addClass('shadow-lg');
});

$(document).on('mouseleave', '.process-step', function() {
    $(this).find('.step-number').removeClass('shadow-lg');
});

// Copy result text to clipboard
$(document).on('click', '[data-copy-result]', function() {
    const text = $(this).data('copy-result');
    Utils.copyToClipboard(text);
});

// Feedback buttons
$(document).on('click', '.btn-outline-success, .btn-outline-danger', function(e) {
    e.preventDefault();
    const isPositive = $(this).hasClass('btn-outline-success');
    const feedback = isPositive ? 'olumlu' : 'olumsuz';
    
    showToast('Teşekkürler', `${feedback} geri bildiriminiz kaydedildi`, 'success', 3000);
    
    // Disable both buttons
    $(this).parent().find('button').prop('disabled', true);
    $(this).removeClass('btn-outline-success btn-outline-danger')
           .addClass(isPositive ? 'btn-success' : 'btn-danger');
});

// Keyboard shortcuts
$(document).keydown(function(e) {
    // Ctrl+Enter or Cmd+Enter to open translate modal
    if ((e.ctrlKey || e.metaKey) && e.which === 13) {
        e.preventDefault();
        $('#translateModal').modal('show');
    }
    
    // Escape to close modals
    if (e.which === 27) {
        $('.modal').modal('hide');
    }
    
    // Ctrl+/ or Cmd+/ for help
    if ((e.ctrlKey || e.metaKey) && e.which === 191) {
        e.preventDefault();
        $('#infoModal').modal('show');
    }
});

// System status check (placeholder)
function checkSystemStatus() {
    // Bu fonksiyon gerçek API endpoint'i olduğunda aktif edilebilir
    $('#system-status').html('<i class="fas fa-circle text-success"></i> Sistem Aktif');
}

// Initialize on load
$(window).on('load', function() {
    // Fade in animations
    $('.feature-card').each(function(index) {
        $(this).delay(index * 100).fadeIn('slow');
    });
    
    // Count up animations for stats
    $('.stat-item h3').each(function() {
        const text = $(this).text();
        if (!isNaN(text)) {
            animateCountUp(this, parseInt(text), 1500);
        }
    });
    
    checkSystemStatus();
});

console.log('📄 main.js yüklendi');
